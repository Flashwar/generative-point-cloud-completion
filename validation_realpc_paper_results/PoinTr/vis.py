import os
import sys
import json
import glob
import argparse
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from easydict import EasyDict

sys.path.append('.')
from datasets.CatenaryDataset import Catenary

#No axes in 3D scene
SCENE_NO_AXES = dict(
    aspectmode='data',
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
)

# load dataset
def load_dataset():
    config = EasyDict({
        'PARTIAL_POINTS_PATH': '/realpc/difficult/%s/partial/%s/%s/%03d.pcd',
        'COMPLETE_POINTS_PATH': '/realpc/difficult/%s/complete/%s/%s.pcd',
        'CATEGORY_FILE_PATH': '/realpc/difficult/ours.json',
        'N_POINTS': 5000, 'N_RENDERINGS': 245, 'subset': 'test', 'CARS': False
    })
    return Catenary(config)

#load points from .npy or .pcd file
def load_points(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    elif path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points, dtype=np.float32)
    return None

#save figures as html 
def save_html(points, color, title, output_path, marker_size=1.5):
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=marker_size, color=color, opacity=0.8),
    )])
    fig.update_layout(
        title=title,
        height=800,
        margin=dict(l=5, r=5, t=60, b=5),
        scene=SCENE_NO_AXES,
        showlegend=False,
    )
    fig.write_html(output_path)
    print(f"  → {output_path}")


def process_sample(idx, run_dir, output_dir, dataset, marker_size=1.5):

    tax_id, model_id, (partial, gt) = dataset[idx]
    short_id = model_id[:8]
    prefix = f"{tax_id}_{short_id}"

    gt_np = gt.numpy()
    partial_np = partial.numpy()

    #save GT 
    save_html(
        gt_np, 'royalblue',
        f'{prefix} — Ground Truth',
        os.path.join(output_dir, f'{prefix}_gt.html'),
        marker_size
    )

    #save Partial Input
    save_html(
        partial_np, 'darkorange',
        f'{prefix} — Partielle Eingabe',
        os.path.join(output_dir, f'{prefix}_partial.html'),
        marker_size
    )

    #save SnowFlakeNet-Prediction
    pred_path = os.path.join(run_dir, '00_snowflake_pred', f'completion_{idx:04d}_snowflake.pcd')
    if os.path.exists(pred_path):
        pred_np = load_points(pred_path)
        if pred_np is not None:
            save_html(
                pred_np, 'forestgreen',
                f'{prefix} — SnowFlakeNet',
                os.path.join(output_dir, f'{prefix}_snowflake.html'),
                marker_size
            )
    else:
        print(f"  SKIP SnowFlakeNet: {pred_path} nicht gefunden")

    #save PPSurf-Resultat
    final_npy = os.path.join(run_dir, '03_final_pointclouds', f'final_{idx:04d}.npy')
    final_pcd = os.path.join(run_dir, '03_final_pointclouds', f'final_{idx:04d}_ppsurf.pcd')

    final_path = final_npy if os.path.exists(final_npy) else (final_pcd if os.path.exists(final_pcd) else None)
    if final_path:
        final_np = load_points(final_path)
        if final_np is not None:
            save_html(
                final_np, 'crimson',
                f'{prefix} — PPSurf',
                os.path.join(output_dir, f'{prefix}_ppsurf.html'),
                marker_size
            )
    else:
        print(f"  SKIP PPSurf: kein Resultat für idx={idx} ({tax_id}) — vermutlich fehlgeschlagen")

    print(f"  [{idx:04d}] {tax_id}/{short_id} fertig")


def main():
    parser = argparse.ArgumentParser(description="Anhang-Visualisierungen: SnowFlakeNet + PPSurf")
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/appendix_figures')
    parser.add_argument('--idx', type=int, nargs='+', default=None,
                        help='Einzelne Sample-Indizes (z.B. --idx 0 5 148)')
    parser.add_argument('--best_worst', action='store_true',
                        help='Alle best/worst Samples aus evaluation_results.json')
    parser.add_argument('--all', action='store_true',
                        help='Alle Samples verarbeiten')
    parser.add_argument('--marker_size', type=float, default=1.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Lade Dataset...")
    dataset = load_dataset()
    print(f"  Dataset-Größe: {len(dataset)}")

    # check which indices to process
    if args.idx:
        indices = args.idx
    elif args.best_worst:
        eval_path = os.path.join(args.run_dir, 'evaluation_results.json')
        if not os.path.exists(eval_path):
            parser.error(f"evaluation_results.json nicht gefunden: {eval_path}")
        with open(eval_path) as f:
            results = json.load(f)
        indices = sorted(set(entry['idx'] for entry in results.get('best_worst', [])))
        print(f"  {len(indices)} best/worst Samples gefunden")
    elif args.all:
        indices = list(range(len(dataset)))
    else:
        parser.error("Bitte --idx, --best_worst oder --all angeben")

    # Samples processing 
    print(f"\nVerarbeite {len(indices)} Samples...\n")
    for idx in indices:
        if idx >= len(dataset):
            print(f"  SKIP idx={idx} (außerhalb Dataset-Größe {len(dataset)})")
            continue
        process_sample(idx, args.run_dir, args.output_dir, dataset, args.marker_size)

    print(f"\nFertig. Alle Dateien unter → {args.output_dir}/")


if __name__ == '__main__':
    main()