#!/usr/bin/env bash

set -euo pipefail

uv run --no-sync python -m viclip_ot.train \
    --run_test_only \
    --seed 42 \
    --model_config ./config/model.jina_clip_v2_lora_clip.yaml \
    --dataset_dir ./data/UIT-OpenViIC \
    --test_split_name test \
    --train_crop_size 512 \
    --eval_resize_size 512 \
    --eval_crop_size 512 \
    --eval_batch_size 32 \
    --num_workers 4 \
    --criterion clip_loss
