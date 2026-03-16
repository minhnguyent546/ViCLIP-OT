#!/usr/bin/env bash

uv run python scripts/download_images_in_dataset.py \
  --dataset_path minhnguyent546/datacomp_large_vie_filtered2 \
  --data_dir ./data/datacomp_large_vie_filtered2 \
  --process_count 16 \
  --thread_count 512 \
  --image_size 448 \
  --num_samples_per_shard 30000 \
  --resize_mode keep_ratio_largest \
  --no_resize_only_if_bigger \
  --encode_format jpg \
  --output_format files \
  --retries 2 \
  --enable_wandb \
  --wandb_project datacomp_vi \
