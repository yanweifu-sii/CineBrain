export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=3600
torchrun --standalone --nproc_per_node=4 train_video_fmri.py \
 --base configs/cogvideox_5b_lora_brain_va.yaml configs/sft_5b_brain_va_clip_sub05.yaml --seed 42
