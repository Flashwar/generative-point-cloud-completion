import torch
import numpy as np
import open3d as o3d
import os
import sys
import json
import subprocess
import trimesh
import argparse
from tqdm import tqdm
from collections import defaultdict
from easydict import EasyDict

sys.path.append('.')
from datasets.CatenaryDataset import Catenary
from models.build import build_model_from_cfg
from extensions.chamfer_dist import ChamferDistanceL1

#All 21 RealPC categories
REALPC_CATEGORIES = [
    'chine_0', 'chine_1', 'chine_2', 'chine_3',
    'dutch_0', 'dutch_1', 'dutch_2', 'dutch_3', 'dutch_4',
    'hung_0', 'hung_1', 'hung_2', 'hung_3', 'hung_4', 'hung_5', 'hung_6', 'hung_7',
    'sncf_0', 'sncf_1', 'sncf_2', 'sncf_3',
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def save_pcd(points, path):
    """Save a numpy point cloud as .pcd file using Open3D"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)


def normalize_pointcloud(points):
    """Center on centroid and scale to unit sphere. Used before CD computation"""
    centroid = np.mean(points, axis=0, keepdims=True)
    pts = points - centroid
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist < 1e-8:
        max_dist = 1.0
    return (pts / max_dist).astype(np.float32)


def add_gaussian_noise(points, sigma_ratio=0.01):
    """
        Add Gaussian noise scaled relative to the bounding box diagonal L
        sigma_ratio=0
        sigma_ratio=0.01 corresponds to medium noise
        sigma_ratio=0.05 corresponds to high noise
        """
    if sigma_ratio <= 0:
        return points.copy()
    L = (points.max(axis=0) - points.min(axis=0)).max()
    return (points + np.random.normal(0, sigma_ratio * L, size=points.shape)).astype(np.float32)


def chamfer_distance_l1l2(pred, gt):
    """Compute both CD-L1 and CD-L2 between two point clouds"""
    with torch.no_grad():
        diff = pred.unsqueeze(1) - gt.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        min_p2g, _ = dist.min(dim=1)
        min_g2p, _ = dist.min(dim=0)
        cd_l1 = 0.5 * (min_p2g.mean() + min_g2p.mean())
        cd_l2 = 0.5 * ((min_p2g ** 2).mean() + (min_g2p ** 2).mean())
    return cd_l1.item(), cd_l2.item()


def run_ppsurf(input_path, output_dir, ppsurf_dir, resolution=129):
    """Run PPSurf surface reconstruction via subprocess"""
    os.makedirs(output_dir, exist_ok=True)
    #seperate venv call
    ppsurf_dir = os.path.abspath(ppsurf_dir)
    ppsurf_python = os.path.join(ppsurf_dir, '.venv_pps', 'bin', 'python')
    cmd = [
        ppsurf_python, 'pps.py', 'rec',
        os.path.abspath(input_path),
        os.path.abspath(output_dir),
        '--model.init_args.gen_resolution_global', str(resolution),
        '--trainer.devices', '1',
    ]
    #PPSurf process call
    try:
        result = subprocess.run(cmd, cwd=ppsurf_dir,
                                capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            err = f"returncode={result.returncode} stderr_tail={result.stderr[-1000:]!r}"
            return None, err
    except subprocess.TimeoutExpired:
        return None, "timeout after 300s"
    except Exception as e:
        return None, f"subprocess exception: {e!r}"

    # PPSurf writes the mesh as .ply inside a subdirectory named after the input file
    input_name = os.path.basename(input_path)
    candidate = os.path.join(output_dir, input_name, input_name + '.ply')
    if os.path.exists(candidate):
        return candidate, None
    #Fallback: search recursively for any .ply file
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.ply'):
                return os.path.join(root, f), None

    return None, f"no .ply found under {output_dir} (cmd exited 0, but no mesh written)"


def mesh_to_pointcloud(mesh_path, n_points=5000):
    """Load a mesh from .ply and sample n_points from its surface"""
    try:
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        return np.asarray(points, dtype=np.float32), None
    except Exception as e:
        return None, f"mesh sampling exception: {e!r}"

def main():
    parser = argparse.ArgumentParser(description="SnowFlakeNet + PPSurf Hybrid Pipeline")
    parser.add_argument('--ppsurf_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/snowflake_ppsurf')
    parser.add_argument('--noise_sigma', type=float, default=0.0,
                        help='0=none, 0.01=medium, 0.05=high')
    parser.add_argument('--ppsurf_resolution', type=int, default=129)
    parser.add_argument('--n_points', type=int, default=5000)
    parser.add_argument('--skip_ppsurf', action='store_true',
                        help='Evaluate SnowFlakeNet alone (no PPSurf)')
    parser.add_argument('--n_best_worst', type=int, default=3,
                        help='How many best/worst samples per category to export as .pcd')
    args = parser.parse_args()

    #Create output subdirectories for each pipeline stage
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir     = os.path.join(args.output_dir, '00_snowflake_pred')
    noisy_dir    = os.path.join(args.output_dir, '01_snowflake_out')
    mesh_dir     = os.path.join(args.output_dir, '02_ppsurf_meshes')
    final_pc_dir = os.path.join(args.output_dir, '03_final_pointclouds')
    showcase_dir = os.path.join(args.output_dir, '04_best_worst')
    for d in [pred_dir, noisy_dir, mesh_dir, final_pc_dir, showcase_dir]:
        os.makedirs(d, exist_ok=True)

    #Step 1: Load Dataset and Model
    print("=" * 60)
    print("Step 1: Loading dataset and model")
    print("=" * 60)

    config = EasyDict({
        'PARTIAL_POINTS_PATH': '/realpc/difficult/%s/partial/%s/%s/%03d.pcd',
        'COMPLETE_POINTS_PATH': '/realpc/difficult/%s/complete/%s/%s.pcd',
        'CATEGORY_FILE_PATH': '/realpc/difficult/ours.json',
        'N_POINTS': 5000, 'N_RENDERINGS': 245, 'subset': 'test', 'CARS': False
    })
    dataset = Catenary(config)

    model_cfg = EasyDict({
        'NAME': 'SnowFlakeNet',
        'dim_feat': 512, 'num_pc': 256, 'num_p0': 512,
        'radius': 1, 'up_factors': [4, 8]
    })
    model = build_model_from_cfg(model_cfg).cuda()
    ckpt = torch.load(
        './experiments/SnowFlakeNet/ours_models/realpc_full_snowflake/ckpt-best.pth',
        map_location='cpu', weights_only=False
    )
    #Remove "module." prefix from state dict keys (artifact of distributed training)
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['base_model'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model. Dataset size: {len(dataset)}")

    #Step 2:SnowFlakeNet Inference
    print("\n" + "=" * 60)
    print("Step 2: SnowFlakeNet inference")
    print("=" * 60)

    #Store metadata and point clouds for each sample
    sample_meta = []
    cd_l1_fn = ChamferDistanceL1()

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Inference"):
            tax_id, model_id, (partial, gt) = dataset[i]
            partial_cuda = partial.unsqueeze(0).cuda()
            gt_cuda = gt.unsqueeze(0).cuda()

            ret = model(partial_cuda)
            pred = ret[-1]  # (1, N, 3)

            #Compute raw CD-L1
            loss = cd_l1_fn(pred, gt_cuda).item() * 1000

            pred_np    = pred[0].cpu().numpy()  # (N, 3)
            gt_np      = gt.numpy()
            partial_np = partial.numpy()

            #Save raw SnowFlakeNet prediction (before noise and PPSurf)
            save_pcd(pred_np, os.path.join(pred_dir, f"completion_{i:04d}_snowflake.pcd"))

            #Apply optional Gaussian noise and save as .npy (input for PPSurf)
            noisy_pred = add_gaussian_noise(pred_np, args.noise_sigma)
            np.save(os.path.join(noisy_dir, f"completion_{i:04d}.npy"), noisy_pred)
            save_pcd(noisy_pred, os.path.join(noisy_dir, f"completion_{i:04d}_noisy.pcd"))

            sample_meta.append({
                'idx': i,
                'tax_id': tax_id,
                'model_id': model_id,
                'gt': gt_np,
                'partial': partial_np,
                'snowflake_pred': pred_np,
                'noisy_pred': noisy_pred,
                'snowflake_cd_l1_raw': loss,
            })

    mean_sfcd = np.mean([s['snowflake_cd_l1_raw'] for s in sample_meta])
    print(f"\nSnowFlakeNet CD-L1 (raw ×1000): {mean_sfcd:.4f}")

    if args.skip_ppsurf:
        #only PPSurf evaluation
        print("\nSkipping PPSurf — evaluating SnowFlakeNet output only")
        final_pcs = [np.load(os.path.join(noisy_dir, f"completion_{i:04d}.npy"))
                     for i in range(len(dataset))]
        failures = []
    else:
        #Step 3 + 4: PPSurf reconstruction and mesh to point cloud
        print("\n" + "=" * 60)
        print(f"Step 3+4: PPSurf (res={args.ppsurf_resolution})")
        print("=" * 60)

        final_pcs = []
        failures = []
        for i in tqdm(range(len(dataset)), desc="PPSurf"):
            input_path  = os.path.abspath(os.path.join(noisy_dir, f"completion_{i:04d}.npy"))
            sample_mesh_dir = os.path.join(mesh_dir, f"completion_{i:04d}")

            mesh_path, err = run_ppsurf(input_path, sample_mesh_dir,
                                        args.ppsurf_dir, args.ppsurf_resolution)
            if mesh_path is None:
                failures.append({
                    'idx': i,
                    'tax_id': sample_meta[i]['tax_id'],
                    'model_id': sample_meta[i]['model_id'],
                    'stage': 'ppsurf',
                    'error': err,
                })
                final_pcs.append(None)
                continue
            #Resample the mesh back to a point cloud
            pc, mesh_err = mesh_to_pointcloud(mesh_path, n_points=args.n_points)
            if pc is not None:
                np.save(os.path.join(final_pc_dir, f"final_{i:04d}.npy"), pc)
                save_pcd(pc, os.path.join(final_pc_dir, f"final_{i:04d}_ppsurf.pcd"))
            else:
                failures.append({
                    'idx': i,
                    'tax_id': sample_meta[i]['tax_id'],
                    'model_id': sample_meta[i]['model_id'],
                    'stage': 'mesh_to_pointcloud',
                    'error': mesh_err,
                })
            final_pcs.append(pc)

        #Print failure summary grouped by category
        if failures:
            print("\n" + "=" * 60)
            print(f"PIPELINE FAILURES: {len(failures)}/{len(dataset)} samples")
            print("=" * 60)
            fail_by_cat = defaultdict(int)
            for fl in failures:
                fail_by_cat[fl['tax_id']] += 1
            for cat, n in sorted(fail_by_cat.items()):
                print(f"  {cat:<12} {n:>3} failures")
            print("\n  First 5 errors:")
            for fl in failures[:5]:
                print(f"    [{fl['idx']:04d}] {fl['tax_id']}/{fl['model_id']} "
                      f"({fl['stage']}): {fl['error']}")

            #Save full failure log as JSON
            failures_path = os.path.join(args.output_dir, 'pipeline_failures.json')
            with open(failures_path, 'w') as f:
                json.dump(failures, f, indent=2, default=str)
            print(f"\n  Full failure log → {failures_path}")

    #Step 5: Evaluation
    print("\n" + "=" * 60)
    print("Step 5: Evaluation (normalized CD-L1 / CD-L2)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cd_l1_list, cd_l2_list = [], []
    per_sample = []

    for i, meta in enumerate(tqdm(sample_meta, desc="CD")):
        meta_light = {k: v for k, v in meta.items()
                      if k not in ('gt', 'partial', 'snowflake_pred', 'noisy_pred')}
        if final_pcs[i] is None:
            per_sample.append({**meta_light, 'cd_l1': None, 'cd_l2': None,
                                'category': meta['tax_id']})
            continue
        #Normalize both prediction and ground truth before computing CD
        pred_norm = normalize_pointcloud(final_pcs[i])
        gt_norm   = normalize_pointcloud(meta['gt'])
        pred_t = torch.from_numpy(pred_norm).float().to(device)
        gt_t   = torch.from_numpy(gt_norm).float().to(device)

        cd_l1, cd_l2 = chamfer_distance_l1l2(pred_t, gt_t)
        cd_l1_list.append(cd_l1)
        cd_l2_list.append(cd_l2)
        per_sample.append({**meta_light, 'cd_l1': cd_l1, 'cd_l2': cd_l2,
                            'category': meta['tax_id']})

    #Print overall results
    valid = len(cd_l1_list)
    print(f"\n{'='*60}")
    print(f"OVERALL  ({valid}/{len(dataset)} samples)")
    print(f"  CD-L1: {np.mean(cd_l1_list):.6f}  (±{np.std(cd_l1_list):.6f})")
    print(f"  CD-L2: {np.mean(cd_l2_list):.6f}  (±{np.std(cd_l2_list):.6f})")

    #Aggregate results per category
    cat_results = defaultdict(lambda: {'cd_l1': [], 'cd_l2': [], 'count': 0})
    for s in per_sample:
        if s['cd_l1'] is None:
            continue
        cat = s['category']
        cat_results[cat]['cd_l1'].append(s['cd_l1'])
        cat_results[cat]['cd_l2'].append(s['cd_l2'])
        cat_results[cat]['count'] += 1

    print(f"\n{'Category':<12} {'N':>4} {'CD-L1':>12} {'CD-L2':>12}")
    print("-" * 44)
    for cat in REALPC_CATEGORIES:
        if cat in cat_results:
            r = cat_results[cat]
            print(f"{cat:<12} {r['count']:>4} "
                  f"{np.mean(r['cd_l1']):>12.6f} {np.mean(r['cd_l2']):>12.6f}")

    super_results = defaultdict(lambda: {'cd_l1': [], 'cd_l2': []})
    for cat, r in cat_results.items():
        sc = cat.split('_')[0]
        super_results[sc]['cd_l1'].extend(r['cd_l1'])
        super_results[sc]['cd_l2'].extend(r['cd_l2'])

    print(f"\n{'Superclass':<12} {'CD-L1':>12} {'CD-L2':>12}")
    print("-" * 38)
    for sc in ['chine', 'dutch', 'hung', 'sncf']:
        if sc in super_results:
            r = super_results[sc]
            print(f"{sc:<12} {np.mean(r['cd_l1']):>12.6f} {np.mean(r['cd_l2']):>12.6f}")

    #Step 6: Best/Worst by categorie
    print("\n" + "=" * 60)
    print(f"Step 6: Exporting best/worst {args.n_best_worst} per category")
    print("=" * 60)


    by_cat = defaultdict(list)
    for i, s in enumerate(per_sample):
        if s['cd_l1'] is not None:
            by_cat[s['category']].append((i, s['cd_l1']))

    exported = []
    for cat, entries in by_cat.items():
        entries.sort(key=lambda x: x[1])
        n = args.n_best_worst
        best_entries  = entries[:n]
        worst_entries = entries[-n:] if len(entries) > n else []
        #Save best samples with all pipeline stages
        for rank, (i, cd_l1) in enumerate(best_entries):
            meta = sample_meta[i]
            prefix = f"{cat}_best_{rank}_cdl1_{cd_l1:.4f}"
            cat_dir = os.path.join(showcase_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)

            save_pcd(meta['partial'],        os.path.join(cat_dir, f"{prefix}_partial.pcd"))
            save_pcd(meta['snowflake_pred'],  os.path.join(cat_dir, f"{prefix}_snowflake.pcd"))
            save_pcd(meta['noisy_pred'],      os.path.join(cat_dir, f"{prefix}_noisy.pcd"))
            if final_pcs[i] is not None:
                save_pcd(final_pcs[i],        os.path.join(cat_dir, f"{prefix}_ppsurf.pcd"))
            save_pcd(meta['gt'],              os.path.join(cat_dir, f"{prefix}_gt.pcd"))
            exported.append({**{k: v for k, v in meta.items()
                                 if k not in ('gt', 'partial', 'snowflake_pred', 'noisy_pred')},
                              'rank': 'best', 'position': rank, 'cd_l1': cd_l1})
        #Save worst samples with all pipeline stages
        for rank, (i, cd_l1) in enumerate(reversed(worst_entries)):
            meta = sample_meta[i]
            prefix = f"{cat}_worst_{rank}_cdl1_{cd_l1:.4f}"
            cat_dir = os.path.join(showcase_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)

            save_pcd(meta['partial'],        os.path.join(cat_dir, f"{prefix}_partial.pcd"))
            save_pcd(meta['snowflake_pred'],  os.path.join(cat_dir, f"{prefix}_snowflake.pcd"))
            save_pcd(meta['noisy_pred'],      os.path.join(cat_dir, f"{prefix}_noisy.pcd"))
            if final_pcs[i] is not None:
                save_pcd(final_pcs[i],        os.path.join(cat_dir, f"{prefix}_ppsurf.pcd"))
            save_pcd(meta['gt'],              os.path.join(cat_dir, f"{prefix}_gt.pcd"))
            exported.append({**{k: v for k, v in meta.items()
                                 if k not in ('gt', 'partial', 'snowflake_pred', 'noisy_pred')},
                              'rank': 'worst', 'position': rank, 'cd_l1': cd_l1})

        print(f"  {cat}: {len(best_entries)} best, {len(worst_entries)} worst → {cat_dir}")

    #Save all results as JSON
    results = {
        'overall': {
            'cd_l1_mean': float(np.mean(cd_l1_list)),
            'cd_l1_std':  float(np.std(cd_l1_list)),
            'cd_l2_mean': float(np.mean(cd_l2_list)),
            'cd_l2_std':  float(np.std(cd_l2_list)),
            'snowflake_cd_l1_raw_mean': float(mean_sfcd),
            'n_evaluated': valid,
            'n_total': len(dataset),
        },
        'config': vars(args),
        'per_category': {
            cat: {
                'count': r['count'],
                'cd_l1': float(np.mean(r['cd_l1'])),
                'cd_l2': float(np.mean(r['cd_l2'])),
            } for cat, r in cat_results.items()
        },
        'per_superclass': {
            sc: {
                'cd_l1': float(np.mean(super_results[sc]['cd_l1'])),
                'cd_l2': float(np.mean(super_results[sc]['cd_l2'])),
            } for sc in ['chine', 'dutch', 'hung', 'sncf'] if sc in super_results
        },
        'per_sample': per_sample,
        'best_worst': exported,
        'failures': failures,
    }

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → {results_path}")
    print("Done!")


if __name__ == '__main__':
    main()