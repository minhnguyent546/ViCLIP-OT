#!/usr/bin/env bash

uv run --no-sync python scripts/compute-embeddings/compute_embeddings_qwen3_vl_embedding.py \
    --model Qwen/Qwen3-VL-Embedding-2B \
    --instruction "Retrieve images or text relevant to the user's query" \
    --dtype bfloat16 \
    --dataset_dir ./data/UIT-OpenViIC \
    --metadata_json_file test.json \
    --batch_size_text 32 \
    --batch_size_image 32 \
    --normalize

# export CUDA_VISIBLE_DEVICES=3

# uv run --no-sync python scripts/compute-embeddings/compute_embeddings_qwen3_vl_embedding.py \
#     --model Qwen/Qwen3-VL-Embedding-8B \
#     --max_pixels 1310720 \
#     --instruction "Retrieve images or text relevant to the user's query" \
#     --backend vllm \
#     --dtype auto \
#     --dataset_dir ./data/mscoco \
#     --metadata_json_file train.qwen3vl32binstruct.json \
#     --batch_size_text 131072 \
#     --batch_size_image 32768 \
#     --normalize \
#     --vllm_max_model_len 8192 \
#     --vllm_gpu_memory_utilization 0.85 \
#     --num_workers 64
#
