# Point Cloud Completion Master thesis



## Project Structure
 
```
generative-point-cloud-completion/
├── Point_Diffusion_Refinement/
│   └── pointnet2/                    # PDR framework (CGNet + RFNet)
│       ├── exp_configs/mvp_configs/  # All configuration files
│       ├── mvp_dataloader/           # Dataloaders (incl. realpc_dataset.py)
│       ├── models/                   # Network architectures
│       ├── train.py                  # Training (CGNet + RFNet)
│       ├── generate_samples.py       # Inference + evaluation
│       └── realpc_dataloader/data/   # Generated data (.h5 files)
├── ppsurf/                           # PPSurf repository
│   ├── pps.py                        # PPSurf CLI entry point
│   └── .venv_pps/                    # Separate Python environment
├── realpc/
│   └── difficult/                    # RealPC dataset
│       ├── train/
│       │   ├── complete/             # GT point clouds (.pcd)
│       │   └── partial/              # Partial scans (.pcd)
│       └── test/
├── hybrid_pipeline.py                # This script
├── results/                          # Evaluation results
└── .pdr-venv/                        # PDR Python environment
```
 
## Prerequisites
 
- Python 3.10
- PyTorch 2.6 + CUDA 12.x
- NVIDIA GPU (tested on H100 NVL)


### PDR Environment Setup
 
```bash
python3.10 -m venv .pdr-venv
source .pdr-venv/bin/activate

#TODO ADDING MISSING DEPENDENCIES

cd Point_Diffusion_Refinement/pointnet2
pip install -e .
```

### PPSurf Environment
 
PPSurf uses a separate environment (`.venv_pps`). See `ppsurf/ppsurf_install.sh`.


## Configuration Files
 
| Config | Purpose |
|--------|---------|
| `config_realpc_completion_5000.json` | CGNet training (from scratch, 5000 points) |
| `config_realpc_refine.json` | RFNet training (on CGNet completions) |
| `config_mvp_on_realpc.json` | Zero-shot: evaluate MVP-CGNet on RealPC |
| `config_realpc_finetune.json` | CGNet fine-tuning (MVP -> RealPC) |
| `config_realpc_finetune_refine.json` | RFNet on fine-tuned CGNet (RFNet from scratch) |
| `config_realpc_finetune_refine_ft.json` | RFNet fine-tuning (MVP-RFNet -> RealPC) |
 
 

## Step-by-step Guide

 
### 1. Train CGNet (~3 days)
 
```bash
cd Point_Diffusion_Refinement/pointnet2
source ../../.pdr-venv/bin/activate
 
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --config exp_configs/mvp_configs/config_realpc_completion_5000.json \
  > training_cgnet.log 2>&1 &
```
 
### 2. Generate CGNet Completions (~3h per set)
 
```bash
# Identify best checkpoint (lowest CD)
grep "Gathered Avg CD loss" training_cgnet.log
 
# Test set completions
CUDA_VISIBLE_DEVICES=0 nohup python generate_samples.py \
  --config exp_configs/mvp_configs/config_realpc_completion_5000.json \
  --ckpt_name pointnet_ckpt_XXXXX.pkl \
  --batch_size 16 --phase test --num_points 5000 --device_ids '0' \
  > generate_test.log 2>&1 &
 
# Training set completions + intermediate state x^100
CUDA_VISIBLE_DEVICES=1 nohup python generate_samples.py \
  --config exp_configs/mvp_configs/config_realpc_completion_5000.json \
  --ckpt_name pointnet_ckpt_XXXXX.pkl \
  --batch_size 16 --phase test_trainset --num_points 5000 \
  --save_multiple_t_slices --t_slices '[100]' --device_ids '0' \
  > generate_train.log 2>&1 &
```
 
### 3. Generate 10 Trials (~3h total)
 
The 10 trials provide diverse training data for the RFNet. Each trial starts from the cached intermediate state x^100 and only runs 100 reverse steps instead of 1000, reducing generation time by 10x.
 
```bash
CUDA_VISIBLE_DEVICES=0 nohup python generate_samples.py \
  --config exp_configs/mvp_configs/config_realpc_completion_5000.json \
  --ckpt_name pointnet_ckpt_XXXXX.pkl \
  --batch_size 16 --phase test_trainset --num_points 5000 \
  --use_a_precomputed_XT --T_step 100 \
  --XT_folder realpc_dataloader/data/realpc_dataset/generated_samples/T1000_betaT0.02_realpc_completion_5000/pointnet_ckpt_XXXXX \
  --num_trials 10 --device_ids '0' \
  > generate_trials.log 2>&1 &
```
 
### 4. Train RFNet (~2-3h)
 
The RFNet refines the coarse CGNet completions by learning per-point displacements. During each epoch, it randomly selects one of the 10 trials for training data diversity.
 
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --config exp_configs/mvp_configs/config_realpc_refine.json \
  > training_rfnet.log 2>&1 &
```
 
### 5. Generate RFNet Completions
 
```bash
# Find best checkpoint
grep "Gathered Avg CD loss" training_rfnet.log
 
CUDA_VISIBLE_DEVICES=0 nohup python generate_samples.py \
  --config exp_configs/mvp_configs/config_realpc_refine.json \
  --ckpt_name pointnet_ckpt_XXXXX_best_cd.pkl \
  --batch_size 16 --phase test --num_points 5000 --device_ids '0' \
  > generate_rfnet.log 2>&1 &
```
 
### 6. Run the Hybrid Pipeline
 
Navigate to the project root
 
#### 6a. Create Subset (193 samples, 1 per object)
 
For fast evaluation, extract every 26th sample (one per object):
 
```bash
python hybrid_pipeline.py --create_subset \
  --pdr_completions Point_Diffusion_Refinement/pointnet2/realpc_dataloader/data/realpc_dataset/generated_samples/T1000_betaT0.02_realpc_completion_5000/refine_exp_ckpt_94149_standard_attention_10_trials/pointnet_ckpt_18829_best_cd/test/realpc_generated_data_5000pts.h5 \
  --subset_output rfnet_subset_193.h5
```
 
#### 6b. Evaluate PDR Only (baseline, completes instantly)
 
```bash
python hybrid_pipeline.py \
  --pdr_completions rfnet_subset_193.h5 \
  --realpc_dir realpc/difficult \
  --ppsurf_dir ppsurf \
  --skip_ppsurf --skip_noise --n_points 5000 \
  --output_dir results/pdr_only
```
 
#### 6c. PDR + PPSurf (various noise levels, ~87 min each)
 
All three can run in parallel on different GPUs:
 
```bash
# No noise
CUDA_VISIBLE_DEVICES=0 nohup python hybrid_pipeline.py \
  --pdr_completions rfnet_subset_193.h5 \
  --realpc_dir realpc/difficult \
  --ppsurf_dir ppsurf \
  --skip_noise --n_points 5000 \
  --output_dir results/hybrid_no_noise \
  > hybrid_none.log 2>&1 &
 
# Moderate noise (sigma = 0.01L)
CUDA_VISIBLE_DEVICES=1 nohup python hybrid_pipeline.py \
  --pdr_completions rfnet_subset_193.h5 \
  --realpc_dir realpc/difficult \
  --ppsurf_dir ppsurf \
  --noise_sigma 0.01 --n_points 5000 \
  --output_dir results/hybrid_noise_0.01 \
  > hybrid_med.log 2>&1 &
 
# Strong noise (sigma = 0.05L)
CUDA_VISIBLE_DEVICES=2 nohup python hybrid_pipeline.py \
  --pdr_completions rfnet_subset_193.h5 \
  --realpc_dir realpc/difficult \
  --ppsurf_dir ppsurf \
  --noise_sigma 0.05 --n_points 5000 \
  --output_dir results/hybrid_noise_0.05 \
  > hybrid_high.log 2>&1 &
```
 
#### 6d. Full Dataset (5018 samples, ~38h)
 
```bash
CUDA_VISIBLE_DEVICES=0 nohup python hybrid_pipeline.py \
  --pdr_completions Point_Diffusion_Refinement/pointnet2/realpc_dataloader/data/realpc_dataset/generated_samples/T1000_betaT0.02_realpc_completion_5000/refine_exp_ckpt_94149_standard_attention_10_trials/pointnet_ckpt_18829_best_cd/test/realpc_generated_data_5000pts.h5 \
  --realpc_dir realpc/difficult \
  --ppsurf_dir ppsurf \
  --skip_noise --n_points 5000 \
  --scans_per_object 26 \
  --output_dir results/hybrid_full_no_noise \
  > hybrid_full.log 2>&1 &
```
 
## Fine-Tuning (MVP -> RealPC)
 
### CGNet Fine-Tuning (~1 day)
 
Uses pre-trained MVP weights as initialization. The `finetune_from` parameter in the config specifies the path to the MVP checkpoint. Layers with matching dimensions are transferred (~99%), mismatched layers (class embedding 16->21, input dimension 4->3) are randomly reinitialized.
 
```bash
cd Point_Diffusion_Refinement/pointnet2
 
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --config exp_configs/mvp_configs/config_realpc_finetune.json \
  > training_finetune.log 2>&1 &
```
 
After training, follow steps 2-5 with the finetune config to generate completions and train the RFNet.
 
### RFNet Fine-Tuning (optional)
 
To also initialize RFNet weights from MVP:
 
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --config exp_configs/mvp_configs/config_realpc_finetune_refine_ft.json \
  > training_rfnet_finetune_ft.log 2>&1 &
```
 
 
## Output Structure
 
```
results/<experiment>/
├── 01_noisy/                    # Noisy point clouds (.npy)
├── 02_ppsurf_meshes/            # PPSurf meshes (.ply)
│   └── completion_XXXX/
│       └── completion_XXXX.npy/
│           └── completion_XXXX.npy.ply
├── 03_final_pointclouds/        # Resampled point clouds (.npy)
├── samples/                     # Per-category samples
└── evaluation_results.json      # All metrics (CD-L1, CD-L2, per category)
```

