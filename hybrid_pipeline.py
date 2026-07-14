import os
import argparse
import numpy as np
import open3d as o3d
import trimesh
import torch
import json
import h5py
import subprocess
from tqdm import tqdm
from collections import defaultdict


#Categories of the realpc Dataset
REALPC_CATEGORIES = [
    'chine_0', 'chine_1', 'chine_2', 'chine_3',
    'dutch_0', 'dutch_1', 'dutch_2', 'dutch_3', 'dutch_4',
    'hung_0', 'hung_1', 'hung_2', 'hung_3', 'hung_4', 'hung_5', 'hung_6', 'hung_7',
    'sncf_0', 'sncf_1', 'sncf_2', 'sncf_3',
]


# Data Loading
def load_pdr_completions(h5_path):
    """Load PDR-generated completions from .h5 file"""
    with h5py.File(h5_path, 'r') as f:
        data = np.array(f['data'])
    print(f"Loaded {data.shape[0]} completions with {data.shape[1]} points from {h5_path}")
    return data


def load_gt_from_realpc(data_dir, split='test', scans_per_object=1):
    """Load GT point clouds from RealPC dataset

    Args:
        data_dir: Root directory of RealPC dataset
        split: 'train' or 'test'
        scans_per_object: How many times to repeat each GT.
            1 = one GT per object (for subset evaluation)
            26 = repeat each GT 26 times (for full evaluation with all scans)
    """

    # read all elements out of the realpc dataset
    complete_dir = os.path.join(data_dir, split, 'complete')
    gt_list = []
    names = []
    labels = []
    objects_count = 0


    for cat in sorted(os.listdir(complete_dir)):
        if cat not in REALPC_CATEGORIES:
            continue
        label = REALPC_CATEGORIES.index(cat)
        cat_dir = os.path.join(complete_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for f in sorted(os.listdir(cat_dir)):
            if not f.endswith('.pcd'):
                continue
            pcd = o3d.io.read_point_cloud(os.path.join(cat_dir, f))
            pts = np.asarray(pcd.points, dtype=np.float32)
            objects_count += 1
            # Repeat GT per scan so that prediction index i
            # corresponds to the same object
            for s in range(scans_per_object):
                gt_list.append(pts)
                if scans_per_object > 1:
                    names.append(f"{cat}/{f.replace('.pcd', '')}/scan_{s:03d}")
                else:
                    names.append(f"{cat}/{f.replace('.pcd', '')}")
                labels.append(label)

    print(f"Loaded {len(gt_list)} GT entries ({objects_count} objects × {scans_per_object} scans) from {complete_dir}")
    return gt_list, names, labels


## Add optional Gaussian noise, following PPSurf convention
def add_gaussian_noise(points, sigma_ratio=0.01):
    """Add Gaussian noise relative to bounding box size L

    PPSurf convention:
        0.0  = no noise
        0.01 = medium noise (0.01L)
        0.05 = high noise (0.05L)

    """
    if sigma_ratio <= 0:
        return points.copy()
    bbox = points.max(axis=0) - points.min(axis=0)
    L = bbox.max()
    sigma = sigma_ratio * L
    noise = np.random.normal(0, sigma, size=points.shape).astype(np.float32)

    #retunr noisy point cloud
    return points + noise


## PPSurf call
def run_ppsurf(input_path, output_dir, ppsurf_dir, resolution=129, device=0):
    """Run PPSurf via subprocess"""

    os.makedirs(output_dir, exist_ok=True)
    # PPSurf uses its own Python environment (.venv_pps)
    ppsurf_dir = os.path.abspath(ppsurf_dir)
    ppsurf_python = os.path.join(
        ppsurf_dir,
        '.venv_pps',
        'bin',
        'python'
    )

    # Pass 'pps.py' instead of full path because PPSurf
    # internally validates sys.argv[0] == 'pps.py'
    ppsurf_script = os.path.join(ppsurf_dir, 'pps.py')
    cmd = [
        ppsurf_python, 'pps.py', 'rec',
        os.path.abspath(input_path),
        os.path.abspath(output_dir),
        '--model.init_args.gen_resolution_global', str(resolution),
        '--trainer.devices', '1',
    ]

    # subprocess call
    try:
        result = subprocess.run(
            cmd, cwd=ppsurf_dir,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"PPSurf error: {result.stderr[-300:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"PPSurf timeout for {input_path}")
        return None

    # Find output mesh
    basename = os.path.splitext(os.path.basename(input_path))[0]
    input_name = os.path.basename(input_path)

    #  PPSurf creates: output_dir/input_name/input_name.ply
    candidate = os.path.join(output_dir, input_name, input_name + '.ply')
    if os.path.exists(candidate):
        return candidate

    # Fallback: recursive search for .ply
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.ply'):
                return os.path.join(root, f)

    print(f"Warning: No mesh found for {input_path}")
    return None

## Converter from mesh to point cloud using area-weighted sampling
def mesh_to_pointcloud(mesh_path, n_points=5000):
    """Sample point cloud from mesh using Trimesh"""
    try:
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        return np.asarray(points, dtype=np.float32)
    except Exception as e:
        print(f"Error sampling mesh {mesh_path}: {e}")
        return None

# Metric
def chamfer_distance(pred, gt):
    """Compute Chamfer Distance (L1 and L2) between two point clouds"""
    with torch.no_grad():
        diff = pred.unsqueeze(1) - gt.unsqueeze(0)  # (N, M, 3)
        dist = torch.norm(diff, dim=-1)  # (N, M)
        min_p2g, _ = torch.min(dist, dim=1)  # (N,) pred->gt
        min_g2p, _ = torch.min(dist, dim=0)  # (M,) gt->pred
        cd_l1 = 0.5 * (torch.mean(min_p2g) + torch.mean(min_g2p))
        cd_l2 = 0.5 * (torch.mean(min_p2g ** 2) + torch.mean(min_g2p ** 2))
    return cd_l1.item(), cd_l2.item()


def normalize_pointcloud(points):
    """Normalize: center and scale to unit sphere"""
    centroid = np.mean(points, axis=0, keepdims=True)
    points_centered = points - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist < 1e-8:
        max_dist = 1.0
    return (points_centered / max_dist).astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Pipeline: PDR + PPSurf")

    # Input
    parser.add_argument('--pdr_completions', type=str, required=True,
                        help='Path to .h5 file with PDR completions')
    parser.add_argument('--realpc_dir', type=str, default=None,
                        help='Path to RealPC dataset root')
    parser.add_argument('--ppsurf_dir', type=str, default=None,
                        help='Path to PPSurf repository')
    parser.add_argument('--output_dir', type=str, default='results/hybrid_pipeline')

    # Pipeline options
    parser.add_argument('--noise_sigma', type=float, default=0.01,
                        help='Noise as fraction of bounding box L (0=none, 0.01=med, 0.05=high)')
    parser.add_argument('--ppsurf_resolution', type=int, default=129)
    parser.add_argument('--n_points', type=int, default=5000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--scans_per_object', type=int, default=1,
                        help='GT repetitions per object (1=subset, 26=full)')

    # Flags
    parser.add_argument('--skip_ppsurf', action='store_true')
    parser.add_argument('--skip_noise', action='store_true')
    parser.add_argument('--compute_1nna', action='store_true')
    parser.add_argument('--save_all_npy', action='store_true',
                        help='Save all samples as .npy (large disk usage)')

    # Subset creation mode
    parser.add_argument('--create_subset', action='store_true',
                        help='Create subset .h5 (1 per object) and exit')
    parser.add_argument('--subset_output', type=str, default='rfnet_subset_193.h5')
    parser.add_argument('--subset_interval', type=int, default=26)

    args = parser.parse_args()


    # Subset creation mode
    # create 193 representative samples (1 per object)
    if args.create_subset:
        with h5py.File(args.pdr_completions, 'r') as f:
            data = np.array(f['data'])
        subset = data[::args.subset_interval]
        with h5py.File(args.subset_output, 'w') as f:
            f.create_dataset('data', data=subset)
        print(f"Created subset: {data.shape} → {subset.shape}, saved to {args.subset_output}")
        return

    # Validation of the arguments
    if args.realpc_dir is None or args.ppsurf_dir is None:
        parser.error("--realpc_dir and --ppsurf_dir are required for pipeline mode")

    os.makedirs(args.output_dir, exist_ok=True)
    # create directories if noise id activated
    noisy_dir = os.path.join(args.output_dir, '01_noisy')
    ppsurf_mesh_dir = os.path.join(args.output_dir, '02_ppsurf_meshes')
    final_pc_dir = os.path.join(args.output_dir, '03_final_pointclouds')
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(ppsurf_mesh_dir, exist_ok=True)
    os.makedirs(final_pc_dir, exist_ok=True)

    # check if the given gpu is available if not cpu
    torch_device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


    # Step 1: Load data
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)

    # Loads PDR completions and GT
    pdr_data = load_pdr_completions(args.pdr_completions)
    gt_list, gt_names, gt_labels = load_gt_from_realpc(
        args.realpc_dir, split='test', scans_per_object=args.scans_per_object)

    # saftey check if the number of predictions matches the GT count
    if len(pdr_data) != len(gt_list):
        print(f"WARNING: PDR completions ({len(pdr_data)}) != GT count ({len(gt_list)})")
        min_count = min(len(pdr_data), len(gt_list))
        pdr_data = pdr_data[:min_count]
        gt_list = gt_list[:min_count]
        gt_names = gt_names[:min_count]
        gt_labels = gt_labels[:min_count]

    n_samples = len(pdr_data)


    # Step 2: Add optional Gaussian Noise
    print("\n" + "=" * 60)
    noise_label = f"σ={args.noise_sigma}L" if not args.skip_noise else "skipped"
    print(f"Step 2: Gaussian noise ({noise_label})")
    print("=" * 60)

    noisy_data = []
    # Each completion is individually noised and saved as .npy
    for i, pc in enumerate(tqdm(pdr_data, desc="Noise")):
        noisy_pc = pc.copy() if args.skip_noise else add_gaussian_noise(pc, sigma_ratio=args.noise_sigma)
        np.save(os.path.join(noisy_dir, f"completion_{i:04d}.npy"), noisy_pc)
        noisy_data.append(noisy_pc)


    #Step 3+4: PPSurf → Mesh → Point Cloud

    #when skip_ppsurf is set, the PDR output is used directly
    if not args.skip_ppsurf:
        print("\n" + "=" * 60)
        print(f"Step 3+4: PPSurf (res={args.ppsurf_resolution}) -> {args.n_points} pts")
        print("=" * 60)

        final_pcs = []
        for i in tqdm(range(n_samples), desc="PPSurf + Mesh -> PC"):
            input_path = os.path.abspath(os.path.join(noisy_dir, f"completion_{i:04d}.npy"))
            sample_out_dir = os.path.abspath(os.path.join(ppsurf_mesh_dir, f"completion_{i:04d}"))

            #PPSurf call
            mesh_path = run_ppsurf(
                input_path, sample_out_dir, args.ppsurf_dir,
                resolution=args.ppsurf_resolution, device=args.device
            )

            # if ppsurf doesn't generate something continue
            if mesh_path is None:
                final_pcs.append(None)
                continue

            #Sample point cloud from mesh
            pc = mesh_to_pointcloud(mesh_path, n_points=args.n_points)
            if pc is not None:
                np.save(os.path.join(final_pc_dir, f"final_{i:04d}.npy"), pc)
            final_pcs.append(pc)
    else:
        print("\n  Skipping PPSurf — evaluating PDR output directly")
        final_pcs = [pc.copy() for pc in noisy_data]


    # Step 5: Evaluation
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)

    cd_l1_list = []
    cd_l2_list = []
    valid_gen_pcs = []
    valid_gt_pcs = []
    per_sample = []

    for i in tqdm(range(n_samples), desc="Computing CD"):
        if final_pcs[i] is None:
            per_sample.append({
                'name': gt_names[i], 'label': gt_labels[i],
                'category': REALPC_CATEGORIES[gt_labels[i]],
                'cd_l1': None, 'cd_l2': None
            })
            continue
        #normalization of both side
        pred_norm = normalize_pointcloud(final_pcs[i])
        gt_norm = normalize_pointcloud(gt_list[i])

        pred_t = torch.from_numpy(pred_norm).float().to(torch_device)
        gt_t = torch.from_numpy(gt_norm).float().to(torch_device)

        cd_l1, cd_l2 = chamfer_distance(pred_t, gt_t)
        cd_l1_list.append(cd_l1)
        cd_l2_list.append(cd_l2)
        valid_gen_pcs.append(pred_norm)
        valid_gt_pcs.append(gt_norm)
        per_sample.append({
            'name': gt_names[i], 'label': gt_labels[i],
            'category': REALPC_CATEGORIES[gt_labels[i]],
            'cd_l1': cd_l1, 'cd_l2': cd_l2
        })

    # Results
    print("\n" + "=" * 60)
    print(f"OVERALL RESULTS ({len(cd_l1_list)}/{n_samples} samples)")
    print("=" * 60)
    print(f"  CD-L1:  {np.mean(cd_l1_list):.6f} (± {np.std(cd_l1_list):.6f})")
    print(f"  CD-L2:  {np.mean(cd_l2_list):.6f} (± {np.std(cd_l2_list):.6f})")

    #Aggregate per category
    cat_results = defaultdict(lambda: {'cd_l1': [], 'cd_l2': [], 'count': 0})
    for s in per_sample:
        if s['cd_l1'] is not None:
            cat = s['category']
            cat_results[cat]['cd_l1'].append(s['cd_l1'])
            cat_results[cat]['cd_l2'].append(s['cd_l2'])
            cat_results[cat]['count'] += 1

    print(f"\n{'Category':<12} {'Count':>6} {'CD-L1':>12} {'CD-L2':>12}")
    print("-" * 46)
    for cat in REALPC_CATEGORIES:
        if cat in cat_results:
            r = cat_results[cat]
            print(f"{cat:<12} {r['count']:>6} {np.mean(r['cd_l1']):>12.6f} {np.mean(r['cd_l2']):>12.6f}")

    # Per-Superclass
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

    print(f"\n{'MEAN':<12} {np.mean(cd_l1_list):>12.6f} {np.mean(cd_l2_list):>12.6f}")


    # Save results
    results = {
        'overall': {
            'cd_l1_mean': float(np.mean(cd_l1_list)),
            'cd_l1_std': float(np.std(cd_l1_list)),
            'cd_l2_mean': float(np.mean(cd_l2_list)),
            'cd_l2_std': float(np.std(cd_l2_list)),
            'n_evaluated': len(cd_l1_list),
            'n_total': n_samples,
        },
        'config': {
            'noise_sigma': args.noise_sigma,
            'skip_noise': args.skip_noise,
            'skip_ppsurf': args.skip_ppsurf,
            'n_points': args.n_points,
            'ppsurf_resolution': args.ppsurf_resolution,
            'scans_per_object': args.scans_per_object,
        },
        'per_category': {},
        'per_superclass': {},
        'per_sample': per_sample,
    }


    for cat in sorted(cat_results.keys()):
        r = cat_results[cat]
        results['per_category'][cat] = {
            'count': r['count'],
            'cd_l1': float(np.mean(r['cd_l1'])),
            'cd_l2': float(np.mean(r['cd_l2'])),
        }

    for sc in ['chine', 'dutch', 'hung', 'sncf']:
        if sc in super_results:
            r = super_results[sc]
            results['per_superclass'][sc] = {
                'cd_l1': float(np.mean(r['cd_l1'])),
                'cd_l2': float(np.mean(r['cd_l2'])),
            }

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Save .npy files
    if args.save_all_npy:
        print("Saving all .npy files...")
        for i in tqdm(range(n_samples), desc="Saving"):
            if final_pcs[i] is None:
                continue
            s = per_sample[i]
            cat = s['category']
            name = s['name'].replace('/', '_')
            sample_dir = os.path.join(args.output_dir, 'samples', cat)
            os.makedirs(sample_dir, exist_ok=True)
            np.save(os.path.join(sample_dir, f"{name}_pred.npy"), final_pcs[i])
            np.save(os.path.join(sample_dir, f"{name}_gt.npy"), gt_list[i])
    else:
        # Save 5 per category
        print("Saving sample .npy files (5 per category)...")
        cat_count = defaultdict(int)
        for i in range(n_samples):
            if final_pcs[i] is None:
                continue
            cat = per_sample[i]['category']
            if cat_count[cat] >= 5:
                continue
            cat_count[cat] += 1
            sample_dir = os.path.join(args.output_dir, 'samples', cat)
            os.makedirs(sample_dir, exist_ok=True)
            name = per_sample[i]['name'].replace('/', '_')
            np.save(os.path.join(sample_dir, f"{name}_pred.npy"), final_pcs[i])
            np.save(os.path.join(sample_dir, f"{name}_gt.npy"), gt_list[i])

    print("\nFinish")


if __name__ == '__main__':
    main()