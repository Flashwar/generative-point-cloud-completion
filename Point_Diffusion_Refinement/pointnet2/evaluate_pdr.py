"""
Evaluate PDR (CGNet + RFNet) on RealPC test set.
Saves partial, CGNet completion, RFNet refinement, and GT for each sample.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import h5py
from tqdm import tqdm

def setup_pdr(pdr_dir):
    """Add PDR to Python path."""
    if pdr_dir not in sys.path:
        sys.path.insert(0, pdr_dir)

def main():
    parser = argparse.ArgumentParser(description="Evaluate PDR and save all intermediate results")
    parser.add_argument('--pdr_dir', type=str, required=True,
                        help='Path to PDR pointnet2/ directory')
    parser.add_argument('--cgnet_config', type=str, required=True,
                        help='Path to CGNet config JSON')
    parser.add_argument('--rfnet_config', type=str, required=True,
                        help='Path to RFNet refine config JSON')
    parser.add_argument('--rfnet_checkpoint', type=str, required=True,
                        help='Path to RFNet checkpoint .pkl')
    parser.add_argument('--cgnet_completions', type=str, required=True,
                        help='Path to CGNet-generated test completions .h5')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Output directory for saved point clouds')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    setup_pdr(args.pdr_dir)

    from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
    from util import print_size
    from dataset import get_dataloader
    from json_reader import restore_string_to_list_in_a_dict

    # ========================================================
    # Load configs
    # ========================================================
    with open(args.rfnet_config) as f:
        config = json.loads(f.read())
    config = restore_string_to_list_in_a_dict(config)

    pointnet_config = config["pointnet_config"]
    pointnet_config['include_t'] = False  # RFNet doesn't use timestep
    refine_config = config["refine_config"]

    # Load CGNet config for dataset
    with open(args.cgnet_config) as f:
        cgnet_config = json.loads(f.read())
    cgnet_config = restore_string_to_list_in_a_dict(cgnet_config)

    if cgnet_config['train_config']['dataset'] == 'realpc_dataset':
        trainset_config = cgnet_config['realpc_dataset_config']
    else:
        raise ValueError(f"Unsupported dataset: {cgnet_config['train_config']['dataset']}")

    # ========================================================
    # Load test data
    # ========================================================
    print("Loading test data...")
    trainset_config['batch_size'] = args.batch_size
    trainset_config['eval_batch_size'] = args.batch_size
    trainset_config['include_generated_samples'] = False
    testloader = get_dataloader(trainset_config, phase='val', rank=0, world_size=1,
                                append_samples_to_last_rank=False)

    # ========================================================
    # Load CGNet completions
    # ========================================================
    print(f"Loading CGNet completions from {args.cgnet_completions}...")
    with h5py.File(args.cgnet_completions, 'r') as f:
        cgnet_data = np.array(f['data'])
    print(f"CGNet completions: {cgnet_data.shape}")

    scale = trainset_config.get('scale', 1)
    # Scale CGNet data to match dataset range
    cgnet_data_scaled = cgnet_data * 2 * scale

    # ========================================================
    # Build and load RFNet
    # ========================================================
    print("Loading RFNet...")
    net = PointNet2CloudCondition(pointnet_config).cuda()
    print_size(net)

    checkpoint = torch.load(args.rfnet_checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.eval()
    print(f"Loaded RFNet from {args.rfnet_checkpoint}")

    output_scale_factor = refine_config['output_scale_factor']

    # ========================================================
    # Run evaluation
    # ========================================================
    print("Running evaluation...")

    all_partial = []
    all_gt = []
    all_cgnet = []
    all_rfnet = []
    all_labels = []

    sample_idx = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(testloader, desc="Evaluating")):
            partial = data['partial'].cuda()
            gt = data['complete'].cuda()
            label = data['label']
            batch_size = gt.shape[0]

            # Get corresponding CGNet completions
            cgnet_batch = torch.from_numpy(
                cgnet_data_scaled[sample_idx:sample_idx + batch_size]
            ).float().cuda()

            # Run RFNet
            net.reset_cond_features()
            displacement = net(cgnet_batch, partial, ts=None, label=label.cuda())
            rfnet_output = cgnet_batch + displacement * output_scale_factor

            # Rescale everything back to original scale
            partial_out = partial / 2 / scale
            gt_out = gt / 2 / scale
            cgnet_out = cgnet_batch / 2 / scale
            rfnet_out = rfnet_output / 2 / scale

            all_partial.append(partial_out.cpu().numpy())
            all_gt.append(gt_out.cpu().numpy())
            all_cgnet.append(cgnet_out.cpu().numpy())
            all_rfnet.append(rfnet_out.cpu().numpy())
            all_labels.append(label.numpy())

            sample_idx += batch_size

    # Concatenate all
    all_partial = np.concatenate(all_partial, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    all_cgnet = np.concatenate(all_cgnet, axis=0)
    all_rfnet = np.concatenate(all_rfnet, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\nTotal samples: {all_partial.shape[0]}")
    print(f"Partial: {all_partial.shape}")
    print(f"CGNet:   {all_cgnet.shape}")
    print(f"RFNet:   {all_rfnet.shape}")
    print(f"GT:      {all_gt.shape}")

    # ========================================================
    # Compute metrics
    # ========================================================
    print("\nComputing metrics...")

    def chamfer_distance_batch(pred, gt):
        """Compute CD-L1 and CD-L2 for a batch."""
        pred_t = torch.from_numpy(pred).float().cuda()
        gt_t = torch.from_numpy(gt).float().cuda()

        cd_l1_list = []
        cd_l2_list = []
        for i in range(pred_t.shape[0]):
            diff = pred_t[i].unsqueeze(0) - gt_t[i].unsqueeze(1)
            dist = torch.norm(diff, dim=-1)
            min_p2g, _ = torch.min(dist, dim=1)
            min_g2p, _ = torch.min(dist, dim=0)
            cd_l1 = 0.5 * (min_p2g.mean() + min_g2p.mean())
            cd_l2 = 0.5 * (torch.mean(min_p2g ** 2) + torch.mean(min_g2p ** 2))
            cd_l1_list.append(cd_l1.item())
            cd_l2_list.append(cd_l2.item())
        return cd_l1_list, cd_l2_list

    # Compute in batches to avoid OOM
    cgnet_cd_l1, cgnet_cd_l2 = [], []
    rfnet_cd_l1, rfnet_cd_l2 = [], []
    bs = 32
    for i in tqdm(range(0, len(all_gt), bs), desc="CD metrics"):
        end = min(i + bs, len(all_gt))
        l1, l2 = chamfer_distance_batch(all_cgnet[i:end], all_gt[i:end])
        cgnet_cd_l1.extend(l1)
        cgnet_cd_l2.extend(l2)
        l1, l2 = chamfer_distance_batch(all_rfnet[i:end], all_gt[i:end])
        rfnet_cd_l1.extend(l1)
        rfnet_cd_l2.extend(l2)

    print(f"\n{'='*50}")
    print(f"{'Metric':<15} {'CGNet':>12} {'RFNet':>12}")
    print(f"{'='*50}")
    print(f"{'CD-L1':<15} {np.mean(cgnet_cd_l1):>12.6f} {np.mean(rfnet_cd_l1):>12.6f}")
    print(f"{'CD-L2':<15} {np.mean(cgnet_cd_l2):>12.6f} {np.mean(rfnet_cd_l2):>12.6f}")
    print(f"{'='*50}")

    # ========================================================
    # Save results
    # ========================================================
    print(f"\nSaving results to {args.output_dir}...")

    # Save as .h5 for compact storage
    h5_path = os.path.join(args.output_dir, 'all_results.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('partial', data=all_partial)
        f.create_dataset('cgnet', data=all_cgnet)
        f.create_dataset('rfnet', data=all_rfnet)
        f.create_dataset('gt', data=all_gt)
        f.create_dataset('labels', data=all_labels)
    print(f"Saved all results to {h5_path}")

    # Save individual samples as .npy for easy viewing
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Save first 20 samples individually
    n_save = min(20, len(all_gt))
    for i in range(n_save):
        sample_dir = os.path.join(samples_dir, f'sample_{i:04d}_label{all_labels[i]}')
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, 'partial.npy'), all_partial[i])
        np.save(os.path.join(sample_dir, 'cgnet.npy'), all_cgnet[i])
        np.save(os.path.join(sample_dir, 'rfnet.npy'), all_rfnet[i])
        np.save(os.path.join(sample_dir, 'gt.npy'), all_gt[i])

    print(f"Saved {n_save} individual samples to {samples_dir}")

    # Save metrics
    metrics = {
        'cgnet_cd_l1_mean': float(np.mean(cgnet_cd_l1)),
        'cgnet_cd_l2_mean': float(np.mean(cgnet_cd_l2)),
        'rfnet_cd_l1_mean': float(np.mean(rfnet_cd_l1)),
        'rfnet_cd_l2_mean': float(np.mean(rfnet_cd_l2)),
        'cgnet_cd_l1_std': float(np.std(cgnet_cd_l1)),
        'rfnet_cd_l1_std': float(np.std(rfnet_cd_l1)),
        'n_samples': len(all_gt),
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()