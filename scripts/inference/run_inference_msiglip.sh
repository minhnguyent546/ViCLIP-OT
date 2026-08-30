#!/usr/bin/env bash

set -euo pipefail

uv run --no-sync python -m viclip_ot.train \
    --run_test_only \
    --seed 42 \
    --model_config ./config/model.msiglip_base_patch16_256_multilingual.yaml \
    --dataset_dir ./data/UIT-OpenViIC \
    --test_split_name test \
    --train_crop_size 256 \
    --eval_resize_size 256 \
    --eval_crop_size 256 \
    --eval_batch_size 32 \
    --num_workers 4 \
    --criterion clip_loss
