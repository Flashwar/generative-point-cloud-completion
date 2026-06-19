"""
Visualize all pipeline steps as interactive HTML files.
Each step is saved as a separate HTML file that can be opened in a browser.

Requires: pip install plotly
"""

import numpy as np
import h5py
import trimesh
import argparse
import os
import sys
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Please install plotly: pip install plotly --break-system-packages")
    sys.exit(1)


def compute_cd(pred, gt):
    """Compute CD-L1."""
    import torch
    pred_t = torch.from_numpy(pred).float()
    gt_t = torch.from_numpy(gt).float()
    diff = pred_t.unsqueeze(0) - gt_t.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    cd = 0.5 * (dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean())
    return cd.item()


def make_pc_html(pts, title, color, output_path, cd_value=None):
    """Save a point cloud as interactive HTML."""
    subtitle = f"CD-L1: {cd_value:.6f}" if cd_value is not None else f"{pts.shape[0]} points"

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=color, opacity=0.7),
    )])

    fig.update_layout(
        title=f"{title}<br><sub>{subtitle}</sub>",
        scene=dict(
            xaxis=dict(range=[-0.7, 0.7], title=''),
            yaxis=dict(range=[-0.7, 0.7], title=''),
            zaxis=dict(range=[-0.7, 0.7], title=''),
            aspectmode='cube',
        ),
        width=800, height=700,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(output_path)


def make_mesh_html(mesh, title, output_path):
    """Save a mesh as interactive HTML."""
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightcoral', opacity=0.7,
        flatshading=True,
    )])

    fig.update_layout(
        title=f"{title}<br><sub>{len(vertices)} vertices, {len(faces)} faces</sub>",
        scene=dict(
            xaxis=dict(range=[-0.7, 0.7], title=''),
            yaxis=dict(range=[-0.7, 0.7], title=''),
            zaxis=dict(range=[-0.7, 0.7], title=''),
            aspectmode='cube',
        ),
        width=800, height=700,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(output_path)


def make_comparison_html(data_dict, title, output_path):
    """Save all steps in one HTML with dropdown selector."""
    fig = go.Figure()

    buttons = []
    for i, (name, info) in enumerate(data_dict.items()):
        visible = (i == 0)

        if info['type'] == 'pointcloud':
            pts = info['data']
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=1.5, color=info['color'], opacity=0.7),
                name=name,
                visible=visible,
            ))
        elif info['type'] == 'mesh':
            mesh = info['data']
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=info['color'], opacity=0.7, flatshading=True,
                name=name,
                visible=visible,
            ))

        visibility = [False] * len(data_dict)
        visibility[i] = True
        buttons.append(dict(
            label=name,
            method='update',
            args=[{'visible': visibility}],
        ))

    fig.update_layout(
        title=title,
        updatemenus=[dict(
            type='dropdown', direction='down',
            x=0.01, y=0.99, xanchor='left', yanchor='top',
            buttons=buttons,
        )],
        scene=dict(
            xaxis=dict(range=[-0.7, 0.7], title=''),
            yaxis=dict(range=[-0.7, 0.7], title=''),
            zaxis=dict(range=[-0.7, 0.7], title=''),
            aspectmode='cube',
        ),
        width=900, height=750,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(output_path)


def visualize_sample(sample_idx, realpc_dir, cgnet_h5, rfnet_h5,
                     ppsurf_mesh_dir, final_pc_dir, pdr_dir, output_dir):
    """Generate all HTML files for a given sample."""

    if pdr_dir not in sys.path:
        sys.path.insert(0, pdr_dir)
    from mvp_dataloader.realpc_dataset import RealPCDataset

    ds = RealPCDataset(
        data_dir=realpc_dir, train=False,
        n_partial=2048, n_complete=5000,
        num_scans_per_object=26, scale=1.0
    )

    # Metadata
    partial_path, complete_path, label = ds.samples[sample_idx]
    parts = partial_path.replace('\\', '/').split('/')
    scan_id = parts[-1].replace('.pcd', '')
    model_id = parts[-2]
    category = parts[-3]

    sample = ds[sample_idx]
    partial = sample['partial'].numpy() / 2
    gt = sample['complete'].numpy() / 2

    # CGNet
    with h5py.File(cgnet_h5, 'r') as f:
        cgnet = np.array(f['data'][sample_idx])

    # RFNet
    with h5py.File(rfnet_h5, 'r') as f:
        rfnet = np.array(f['data'][sample_idx])

    # PPSurf mesh
    mesh = None
    for idx in [sample_idx]:
        mesh_dir = os.path.join(ppsurf_mesh_dir, f"completion_{idx:04d}")
        if os.path.isdir(mesh_dir):
            for root, dirs, files in os.walk(mesh_dir):
                for f in files:
                    if f.endswith('.ply'):
                        try:
                            mesh = trimesh.load(os.path.join(root, f))
                            if isinstance(mesh, trimesh.Scene):
                                mesh = mesh.dump(concatenate=True)
                        except:
                            pass
                        break

    # Final PC
    final_pc = None
    final_path = os.path.join(final_pc_dir, f"final_{sample_idx:04d}.npy")
    if os.path.exists(final_path):
        final_pc = np.load(final_path)

    # Create output directory
    sample_dir = os.path.join(output_dir, f"{category}_{model_id}_{scan_id}")
    os.makedirs(sample_dir, exist_ok=True)

    prefix = f"{category} / {model_id} / {scan_id}"

    # CD values
    cd_cgnet = compute_cd(cgnet, gt)
    cd_rfnet = compute_cd(rfnet, gt)
    cd_final = compute_cd(final_pc, gt) if final_pc is not None else None

    # 1. Partial
    make_pc_html(partial, f"1 — Partial Input | {prefix}",
                 'gray', os.path.join(sample_dir, '1_partial.html'))

    # 2. CGNet
    make_pc_html(cgnet, f"2 — CGNet (Coarse) | {prefix}",
                 'orange', os.path.join(sample_dir, '2_cgnet.html'), cd_cgnet)

    # 3. RFNet
    make_pc_html(rfnet, f"3 — RFNet (Refined) | {prefix}",
                 'dodgerblue', os.path.join(sample_dir, '3_rfnet.html'), cd_rfnet)

    # 4. PPSurf Mesh
    if mesh is not None:
        make_mesh_html(mesh, f"4 — PPSurf Mesh | {prefix}",
                       os.path.join(sample_dir, '4_ppsurf_mesh.html'))

    # 5. Final PC
    if final_pc is not None:
        make_pc_html(final_pc, f"5 — Final PC (Resampled) | {prefix}",
                     'green', os.path.join(sample_dir, '5_final_pc.html'), cd_final)

    # 6. GT
    make_pc_html(gt, f"6 — Ground Truth | {prefix}",
                 'crimson', os.path.join(sample_dir, '6_gt.html'))

    # 7. Combined (dropdown)
    data_dict = {
        f'Partial ({partial.shape[0]} pts)': {'type': 'pointcloud', 'data': partial, 'color': 'gray'},
        f'CGNet (CD: {cd_cgnet:.4f})': {'type': 'pointcloud', 'data': cgnet, 'color': 'orange'},
        f'RFNet (CD: {cd_rfnet:.4f})': {'type': 'pointcloud', 'data': rfnet, 'color': 'dodgerblue'},
    }
    if mesh is not None:
        data_dict['PPSurf Mesh'] = {'type': 'mesh', 'data': mesh, 'color': 'lightcoral'}
    if final_pc is not None:
        data_dict[f'Final PC (CD: {cd_final:.4f})'] = {'type': 'pointcloud', 'data': final_pc, 'color': 'green'}
    data_dict[f'Ground Truth ({gt.shape[0]} pts)'] = {'type': 'pointcloud', 'data': gt, 'color': 'crimson'}

    make_comparison_html(data_dict, f"Pipeline — {prefix}",
                         os.path.join(sample_dir, '0_comparison.html'))

    # Summary
    summary = {
        'sample_idx': sample_idx,
        'category': category, 'model_id': model_id, 'scan_id': scan_id,
        'cd_cgnet': cd_cgnet, 'cd_rfnet': cd_rfnet, 'cd_final': cd_final,
        'has_mesh': mesh is not None, 'has_final_pc': final_pc is not None,
    }
    with open(os.path.join(sample_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [{category}] {model_id}/{scan_id}: "
          f"CGNet={cd_cgnet:.4f}, RFNet={cd_rfnet:.4f}"
          + (f", Final={cd_final:.4f}" if cd_final else ""))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdr_dir', type=str,
                        default='Point_Diffusion_Refinement/pointnet2')
    parser.add_argument('--realpc_dir', type=str, default='realpc/difficult')
    parser.add_argument('--cgnet_h5', type=str,
                        default='Point_Diffusion_Refinement/pointnet2/realpc_dataloader/data/realpc_dataset/generated_samples/T1000_betaT0.02_realpc_completion_5000/pointnet_ckpt_94149/test/realpc_generated_data_5000pts.h5')
    parser.add_argument('--rfnet_h5', type=str,
                        default='Point_Diffusion_Refinement/pointnet2/realpc_dataloader/data/realpc_dataset/generated_samples/T1000_betaT0.02_realpc_completion_5000/refine_exp_ckpt_94149_standard_attention_10_trials/pointnet_ckpt_24209_best_cd/test/realpc_generated_data_5000pts.h5')
    parser.add_argument('--ppsurf_mesh_dir', type=str,
                        default='results/hybrid_full_no_noise/02_ppsurf_meshes')
    parser.add_argument('--final_pc_dir', type=str,
                        default='results/hybrid_full_no_noise/03_final_pointclouds')
    parser.add_argument('--output_dir', type=str, default='results/visualizations')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Single sample index')
    parser.add_argument('--n_per_category', type=int, default=3,
                        help='Samples per category for --all')
    parser.add_argument('--all', action='store_true',
                        help='Visualize samples from all categories')
    args = parser.parse_args()

    if args.sample_idx is not None:
        visualize_sample(
            args.sample_idx, args.realpc_dir, args.cgnet_h5, args.rfnet_h5,
            args.ppsurf_mesh_dir, args.final_pc_dir, args.pdr_dir, args.output_dir
        )
    elif args.all:
        sys.path.insert(0, args.pdr_dir)
        from mvp_dataloader.realpc_dataset import RealPCDataset, REALPC_CATEGORIES

        ds = RealPCDataset(
            data_dir=args.realpc_dir, train=False,
            n_partial=2048, n_complete=5000,
            num_scans_per_object=26, scale=1.0
        )

        cat_samples = {}
        for i, (_, _, label) in enumerate(ds.samples):
            cat = REALPC_CATEGORIES[label]
            if cat not in cat_samples:
                cat_samples[cat] = []
            if len(cat_samples[cat]) < args.n_per_category:
                cat_samples[cat].append(i)

        for cat in sorted(cat_samples.keys()):
            print(f"\n=== {cat} ===")
            for idx in cat_samples[cat]:
                try:
                    visualize_sample(
                        idx, args.realpc_dir, args.cgnet_h5, args.rfnet_h5,
                        args.ppsurf_mesh_dir, args.final_pc_dir, args.pdr_dir,
                        args.output_dir
                    )
                except Exception as e:
                    print(f"  Error for sample {idx}: {e}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()