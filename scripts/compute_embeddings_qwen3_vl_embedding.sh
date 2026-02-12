#!/usr/bin/env bash

python scripts/compute_embeddings_qwen3_vl_embedding.py \
    --model Qwen/Qwen3-VL-Embedding-2B \
    --instruction "Retrieve images or text relevant to the user's query" \
    --dtype bfloat16 \
    --dataset_dir ./data/UIT-OpenViIC \
    --metadata_json_file test.json \
    --batch_size 32 \
    --normalize