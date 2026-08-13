#!/usr/bin/env bash

# export WANDB_API_KEY='<YOUR_WANDB_API_KEY_HERE>'

uv run --no-sync python -m viclip_ot.train \
  --optimizer adamw \
  --adam_eps 1e-10 \
  --seed 42 \
  --model_config ./config/model.vit_base_dinov3_sbert_siglip.yaml \
  --dataset_dir ./data/UIT-OpenViIC \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --train_crop_size 224 \
  --eval_resize_size 256 \
  --eval_crop_size 224 \
  --checkpoints_dir ./checkpoints \
  --num_epochs 30 \
  --num_workers 8 \
  --log_file_interval 3 \
  --mixed_precision bf16 \
  --gradient_accum_steps 8 \
  --lr 2e-4 \
  --backbone_lr 5e-5 \
  --lock_image \
  --lock_image_last_unfreeze_groups 14 \
  --weight_decay 1e-4 \
  --scheduler one_cycle_lr \
  --min_lr 1e-6 \
  --lr_warmup_epochs 2 \
  --lr_warmup_method linear \
  --best_checkpoint_metrics t2i_R__1 \
  --save_best_k 5 \
  --max_grad_norm 1.0 \
  --wandb_logging \
  --wandb_project viclip_ot \
  --wandb_name siglip \
  --criterion sig_lip_loss
