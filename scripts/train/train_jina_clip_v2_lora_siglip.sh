#!/usr/bin/env bash


uv run --no-sync python -m viclip_ot.train \
  --optimizer adamw \
  --adam_eps 1e-10 \
  --seed 42 \
  --model_config ./config/model.jina_clip_v2_lora_siglip.yaml \
  --dataset_dir ./data/UIT-OpenViIC \
  --train_batch_size 8 \
  --gradient_accum_steps 16 \
  --eval_batch_size 2 \
  --train_crop_size 512 \
  --eval_resize_size 512 \
  --eval_crop_size 512 \
  --checkpoints_dir ./checkpoints \
  --num_epochs 15 \
  --num_workers 4 \
  --log_file_interval 3 \
  --mixed_precision bf16 \
  --lr 2e-4 \
  --backbone_lr 2e-4 \
  --weight_decay 1e-4 \
  --scheduler one_cycle_lr \
  --min_lr 1e-6 \
  --lr_warmup_epochs 2 \
  --lr_warmup_method linear \
  --best_checkpoint_metrics t2i_R__1 \
  --save_best_k 5 \
  --save_best_k_only \
  --max_grad_norm 1.0 \
  --wandb_logging \
  --wandb_project viclip_ot \
  --wandb_name jina_clip_v2_lora_r16_a32_siglip \
  --criterion sig_lip_loss
