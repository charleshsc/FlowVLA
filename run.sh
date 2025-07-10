export LIBGL_ALWAYS_SOFTWARE=1

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0  scripts/train_diffusion_oxe.py \
    task_name=finetuning_LIBERO dataname=libero_spatial_no_noops dataset.traj_length=11 \
    dataset.data_path=/ailab/user/group/medai-share/hsc_dataset/Libero \
    num_pred_action=10 scheduler_type=0 shuffle_buffer_size=128000 use_close_loop_eval=True \
    +trajectory_dim=7 +pretrained_path=dit_policy_checkpoint.pth +use_adjust_scheduler=true \
    lr=0.0001 +min_lr_scale=0.01 +image_aug=true num_inference_steps=10