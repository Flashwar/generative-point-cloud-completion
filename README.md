# Point Cloud Completion Master thesis




## starting the Pipeline

### create a Subset (193 objects)
python hybrid_pipeline.py --create_subset \
  --pdr_completions Point_Diffusion_Refinement/pointnet2/realpc_dataloader/data/realpc_dataset/generated_samples/T1000_betaT0.02_realpc_completion_5000/refine_exp_ckpt_94149_standard_attention_10_trials/pointnet_ckpt_24209_best_cd/test/realpc_generated_data_5000pts.h5 \
  --subset_output rfnet_subset_193.h5


CUDA_VISIBLE_DEVICES=0 nohup python hybrid_pipeline.py \
  --pdr_completions rfnet_subset_193.h5 \
  --realpc_dir realpc/difficult \
  --ppsurf_dir ppsurf \
  --noise_sigma 0.01 --n_points 5000 \
  --output_dir results/hybrid_noise_0.01 \
  > hybrid_med.log 2>&1 &

