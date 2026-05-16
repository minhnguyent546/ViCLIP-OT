#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR='./data/UIT-OpenViIC'
METADATA_JSON_FILE='val.json'
OUTPUT_CSV='latency_results.csv'
DEVICE='cuda'
WARMUP_IMAGES=100
WARMUP_CAPTIONS=100
DTYPE='bfloat16'

# ViCLIP-OT
uv run python scripts/measure_latency.py \
    --model_family 'viclip-ot' \
    --model_name 'minhnguyent546/ViCLIP-OT' \
    --dataset_dir "$DATASET_DIR" \
    --metadata_json_file "$METADATA_JSON_FILE" \
    --output_csv "$OUTPUT_CSV" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch_size 32 \
    --warmup_images "$WARMUP_IMAGES" \
    --warmup_captions "$WARMUP_CAPTIONS"

# mSigLIP-base
uv run python scripts/measure_latency.py \
    --model_family 'msiglip' \
    --model_name 'google/siglip-base-patch16-256-multilingual' \
    --dataset_dir "$DATASET_DIR" \
    --metadata_json_file "$METADATA_JSON_FILE" \
    --output_csv "$OUTPUT_CSV" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch_size 32 \
    --warmup_images "$WARMUP_IMAGES" \
    --warmup_captions "$WARMUP_CAPTIONS"

# nllb-clip-large-siglip
uv run python scripts/measure_latency.py \
    --model_family 'nllb-clip' \
    --model_name 'nllb-clip-large-siglip:v1' \
    --dataset_dir "$DATASET_DIR" \
    --metadata_json_file "$METADATA_JSON_FILE" \
    --output_csv "$OUTPUT_CSV" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch_size 32 \
    --warmup_images "$WARMUP_IMAGES" \
    --warmup_captions "$WARMUP_CAPTIONS"

# jina-clip-v2 (jina-clip-v2 uses FA2 by default)
uv run python scripts/measure_latency.py \
    --model_family 'jina-clip-v2' \
    --model_name 'jinaai/jina-clip-v2' \
    --dataset_dir "$DATASET_DIR" \
    --metadata_json_file "$METADATA_JSON_FILE" \
    --output_csv "$OUTPUT_CSV" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch_size 32 \
    --warmup_images "$WARMUP_IMAGES" \
    --warmup_captions "$WARMUP_CAPTIONS"

# jina-embeddings-v4 (jina-embeddings-v4 uses FA2 by default)
uv run python scripts/measure_latency.py \
    --model_family 'jina-embeddings-v4' \
    --model_name 'jinaai/jina-embeddings-v4' \
    --dataset_dir "$DATASET_DIR" \
    --metadata_json_file "$METADATA_JSON_FILE" \
    --output_csv "$OUTPUT_CSV" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch_size 32 \
    --warmup_images "$WARMUP_IMAGES" \
    --warmup_captions "$WARMUP_CAPTIONS"

# Qwen3-VL-Embedding-2B
uv run python scripts/measure_latency.py \
    --model_family 'qwen3-vl-embedding' \
    --model_name 'Qwen/Qwen3-VL-Embedding-2B' \
    --dataset_dir "$DATASET_DIR" \
    --metadata_json_file "$METADATA_JSON_FILE" \
    --output_csv "$OUTPUT_CSV" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch_size 32 \
    --warmup_images "$WARMUP_IMAGES" \
    --warmup_captions "$WARMUP_CAPTIONS" \
    --use_flash_attn
