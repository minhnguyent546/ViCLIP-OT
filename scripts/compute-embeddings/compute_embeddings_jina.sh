#!/usr/bin/env bash

uv run --no-sync python scripts/compute-embeddings/compute_embeddings_jina.py \
    --model jinaai/jina-embeddings-v4 \
    --dtype bfloat16 \
    --dataset_dir ./data/UIT-OpenViIC \
    --metadata_json_file test.json \
    --batch_size 32 \
    --normalize \
    --use_flash_attn