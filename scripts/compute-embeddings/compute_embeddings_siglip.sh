#!/usr/bin/env bash

uv run --no-sync python scripts/compute-embeddings/compute_embeddings_siglip.py \
    --model google/siglip-base-patch16-256-multilingual \
    --dtype float32 \
    --dataset_dir ./data/UIT-OpenViIC \
    --metadata_json_file test.json \
    --batch_size 32 \
    --normalize