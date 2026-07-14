import os
import sys
import re
import json
import argparse
import numpy as np
import plotly.graph_objects as go
from easydict import EasyDict
from plotly.subplots import make_subplots

sys.path.append('.')
from datasets.CatenaryDataset import Catenary


# load dataset
def load_dataset():
    config = EasyDict({
        'PARTIAL_POINTS_PATH': '/home/tepper/base/realpc/difficult/%s/partial/%s/%s/%03d.pcd',
        'COMPLETE_POINTS_PATH': '/home/tepper/base/realpc/difficult/%s/complete/%s/%s.pcd',
        'CATEGORY_FILE_PATH': '/home/tepper/base/realpc/difficult/ours.json',
        'N_POINTS': 5000, 'N_RENDERINGS': 245, 'subset': 'test', 'CARS': False
    })
    return Catenary(config)

#load name of failed samples from pipeline_failures.json and check if corresponding .npy files exist
def collect_samples_from_json(failures_json, npy_dir):
    with open(failures_json, 'r') as f:
        failures = json.load(f)

    samples = []
    for entry in failures:
        idx = entry['idx']
        tax_id = entry.get('tax_id', '?')
        model_id = entry.get('model_id', '?')
        npy_path = os.path.join(npy_dir, f"completion_{idx:04d}.npy")
        if not os.path.exists(npy_path):
            print(f"  WARNING: erwartete Datei fehlt: {npy_path}")
            continue
        samples.append({'idx': idx, 'tax_id': tax_id, 'model_id': model_id, 'npy_path': npy_path})
    return samples


def collect_samples_from_dir(npy_dir):
    """Fallback: parse 'completion_XXXX' Indices out of the .npy filenames in the given directory"""
    pattern = re.compile(r'completion_(\d{4})')
    samples = []
    for fname in sorted(os.listdir(npy_dir)):
        if not fname.endswith('.npy'):
            continue
        m = pattern.search(fname)
        if not m:
            continue
        idx = int(m.group(1))
        samples.append({
            'idx': idx, 'tax_id': '?', 'model_id': '?',
            'npy_path': os.path.join(npy_dir, fname)
        })
    return samples

#Create a Scatter3D trace for Plotly
def make_scatter3d(points, color, name):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=color, opacity=0.8),
        name=name,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualisiert fehlgeschlagene Samples (Input vs. GT)")
    parser.add_argument('--failures_json', type=str, default=None,
                        help='Pfad zu pipeline_failures.json (bevorzugt, liefert tax_id/model_id)')
    parser.add_argument('--npy_dir', type=str, required=True,
                        help='Ordner mit den completion_XXXX.npy Input-Dateien')
    parser.add_argument('--output_html', type=str, default='./failed_samples_viewer.html')
    args = parser.parse_args()

    print("Lade Dataset (für GT-Punktwolken)...")
    dataset = load_dataset()
    print(f"  Dataset-Größe: {len(dataset)}")

    if args.failures_json and os.path.exists(args.failures_json):
        print(f"Lese Sample-Liste aus {args.failures_json}")
        samples = collect_samples_from_json(args.failures_json, args.npy_dir)
    else:
        print(f"Keine failures_json angegeben/gefunden — parse Dateinamen aus {args.npy_dir}")
        samples = collect_samples_from_dir(args.npy_dir)

    if not samples:
        print("Keine Samples gefunden — Abbruch.")
        return

    print(f"{len(samples)} Samples werden geladen...")

    traces = []
    labels = []

    for s in samples:
        idx = s['idx']
        input_pc = np.load(s['npy_path']).astype(np.float32)

        # Load GT point cloud from dataset
        _, _, (_, gt) = dataset[idx]
        gt_pc = gt.numpy().astype(np.float32)

        # Create Scatter3D traces for input and GT
        input_trace = make_scatter3d(input_pc, 'crimson', 'Input (SnowFlakeNet-Output)')
        gt_trace = make_scatter3d(gt_pc, 'royalblue', 'Ground Truth')

        traces.append((input_trace, gt_trace))
        labels.append(f"[{idx:04d}] {s['tax_id']} ({s['model_id'][:8]})")

    n = len(samples)
    print(f"Baue Figure mit {n} Sample-Paaren (2 Subplots: Input | GT)...")


    # Create a subplot figure with two 3D scenes (Input and GT) and a dropdown to switch between samples
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Input (SnowFlakeNet-Output)', 'Ground Truth'),
        horizontal_spacing=0.02,
    )

    # Add traces for the first sample and set visibility for others to False
    for i, (input_trace, gt_trace) in enumerate(traces):
        input_trace.visible = (i == 0)
        gt_trace.visible = (i == 0)
        fig.add_trace(input_trace, row=1, col=1)
        fig.add_trace(gt_trace, row=1, col=2)

    buttons = []
    for i, label in enumerate(labels):
        visibility = [False] * (2 * n)
        visibility[2 * i] = True
        visibility[2 * i + 1] = True
        buttons.append(dict(
            label=label,
            method='update',
            args=[{'visible': visibility},
                  {'title': f"Fehlgeschlagenes Sample: {label}"}]
        ))

    fig.update_layout(
        title=f"Fehlgeschlagenes Sample: {labels[0]}",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.5, xanchor='center',
            y=1.12, yanchor='top',
            direction='down',
            showactive=True,
        )],
        height=700,
        margin=dict(l=10, r=10, t=120, b=10),
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
    )

    # Save the figure as an HTML file
    os.makedirs(os.path.dirname(os.path.abspath(args.output_html)) or '.', exist_ok=True)
    fig.write_html(args.output_html)
    print(f"\nGespeichert → {args.output_html}")
    print(f"({n} Samples, durchschaltbar über Dropdown oben)")


if __name__ == '__main__':
    main()