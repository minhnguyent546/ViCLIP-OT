#!/usr/bin/env bash

uv run --no-sync python scripts/compute-embeddings/compute_embeddings_nllb_clip.py \
    --model nllb-clip-large-siglip:v1 \
    --language vie_Latn \
    --dataset_dir ./data/UIT-OpenViIC \
    --metadata_json_file test.json \
    --batch_size 64 \
    --normalize
