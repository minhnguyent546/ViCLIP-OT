#!/usr/bin/env bash

set -euo pipefail

# download pre-trained checkpoint
if [[ ! -f ./checkpoints/viclip_ot/viclip_ot.pth ]]; then
uvx hf download minhnguyent546/ViCLIP-OT-checkpoints \
  --local-dir checkpoints \
  --include viclip_ot/viclip_ot.pth
fi

uv run --no-sync python -m viclip_ot.train \
  --run_test_only \
  --from_checkpoint ./checkpoints/viclip_ot/viclip_ot.pth \
  --seed 42 \
  --model_config ./config/model.vit_base_dinov3_sbert.yaml \
  --dataset_dir ./data/UIT-OpenViIC \
  --eval_batch_size 32 \
  --eval_resize_size 256 \
  --eval_crop_size 224 \
  --num_workers 4
