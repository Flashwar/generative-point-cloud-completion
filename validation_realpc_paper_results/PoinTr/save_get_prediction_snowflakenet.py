import torch
import numpy as np
import open3d as o3d
import os
import sys

sys.path.append('.')
from datasets.CatenaryDataset import Catenary
from easydict import EasyDict
from models.build import build_model_from_cfg
from extensions.chamfer_dist import ChamferDistanceL1

def save_pcd(points, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

output_dir = "./vis_results/SnowFlakeNet"
os.makedirs(output_dir, exist_ok=True)

# Dataset
config = EasyDict({
    'PARTIAL_POINTS_PATH': '/realpc/difficult/%s/partial/%s/%s/%03d.pcd',
    'COMPLETE_POINTS_PATH': '/realpc/difficult/%s/complete/%s/%s.pcd',
    'CATEGORY_FILE_PATH': '/realpc/difficult/ours.json',
    'N_POINTS': 5000, 'N_RENDERINGS': 245, 'subset': 'test', 'CARS': False
})
dataset = Catenary(config)

# Model
model_cfg = EasyDict({
    'NAME': 'SnowFlakeNet',
    'dim_feat': 512,
    'num_pc': 256,
    'num_p0': 512,
    'radius': 1,
    'up_factors': [4, 8]
})


model = build_model_from_cfg(model_cfg).cuda()
ckpt = torch.load('./experiments/SnowFlakeNet/ours_models/realpc_full_snowflake/ckpt-best.pth', map_location='cpu')
state_dict = ckpt['base_model']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

cd_l1 = ChamferDistanceL1()

results = []
with torch.no_grad():
    for i in range(len(dataset)):
        tax_id, model_id, (partial, gt) = dataset[i]
        partial_cuda = partial.unsqueeze(0).cuda()
        gt_cuda = gt.unsqueeze(0).cuda()

        ret = model(partial_cuda)
        pred = ret[-1]  # dense prediction

        loss = cd_l1(pred, gt_cuda).item() * 1000
        results.append((i, tax_id, model_id, loss))
        print(f"[{i+1}/{len(dataset)}] {tax_id}/{model_id}: CD-L1 = {loss:.2f}")

# Sortiere nach Loss
results.sort(key=lambda x: x[3])

print("\n=== BEST 5 ===")
for i, tax, mid, loss in results[:5]:
    print(f"  {tax}/{mid}: CD-L1 = {loss:.2f}")

print("\n=== WORST 5 ===")
for i, tax, mid, loss in results[-5:]:
    print(f"  {tax}/{mid}: CD-L1 = {loss:.2f}")

# Speichere best und worst
for label, items in [("best", results[:3]), ("worst", results[-3:])]:
    for idx, (i, tax, mid, loss) in enumerate(items):
        _, _, (partial, gt) = dataset[i]
        partial_cuda = partial.unsqueeze(0).cuda()
        with torch.no_grad():
            ret = model(partial_cuda)
            pred = ret[-1]

        prefix = f"{output_dir}/{label}_{idx}_{tax}_{mid}"
        save_pcd(partial.numpy(), f"{prefix}_partial.pcd")
        save_pcd(gt.numpy(), f"{prefix}_gt.pcd")
        save_pcd(pred[0].cpu().numpy(), f"{prefix}_pred.pcd")
        print(f"Saved {prefix} (CD-L1: {loss:.2f})")