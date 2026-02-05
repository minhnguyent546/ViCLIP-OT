#!/usr/bin/env bash

python scripts/compute_embeddings_jina_embedding_v4.py \
    --model jinaai/jina-embeddings-v4 \
    --dtype bfloat16 \
    --dataset_dir ./data/UIT-OpenViIC \
    --metadata_json_file test.json \
    --batch_size 32 \
    --normalize