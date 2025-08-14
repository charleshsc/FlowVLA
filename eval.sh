# export MUJOCO_GL=osmesa
# export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1

CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --rdzv_endpoint=localhost:29400  scripts/train_diffusion_oxe.py \
    dataset.traj_length=11 \
    dataset.data_path=/ailab/user/group/medai-share/hsc_dataset/Libero \
    num_pred_action=10 \
    scheduler_type=5 \
    shuffle_buffer_size=128000 \
    dataname=libero_object_no_noops \
    task_name=finetuning_LIBERO \
    +clamp_value=False \
    +max_action=2.5 \
    batch_size=64 \
    +use_adjust_scheduler=True \
    +min_lr_scale=0.05 \
    epoch=20  +eval_only=1 \
    use_close_loop_eval=True +trajectory_dim=7 \
    ckpt_path=/ailab/user/zhouxiao/hsc/VLA/Dita/outputs/2025-07-24/02-20-46/checkpoints/ckpt_20000.pth