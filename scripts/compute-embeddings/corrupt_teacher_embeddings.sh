#!/usr/bin/env bash
# Corrupt the Qwen3-VL-Embedding-2B teacher embeddings for the SIGROT
# graph-quality ablation.
#
# Grid: noise rho* in {0.9, 0.7, 0.5} + random (structureless control).
# rho*=0.3 is intentionally absent: full temperature absorption would need
# logit_scale ~ 200+, beyond the learned-scale clamp of 100.
# All other training settings stay identical to the clean baseline run.

set -euo pipefail

EMB_DIR="data/UIT-OpenViIC-embeddings"
IMAGE_EMB="${EMB_DIR}/train_image_embeddings_qwen3_vl_embedding_2b.pt"
CAPTION_EMB="${EMB_DIR}/train_caption_embeddings_qwen3_vl_embedding_2b.pt"
SEED=42

for RHO_STAR in 0.9 0.7 0.5; do
    uv run --no-sync python scripts/compute-embeddings/corrupt_teacher_embeddings.py \
        --mode noise \
        --rho_star "${RHO_STAR}" \
        --seed "${SEED}" \
        --image_embeddings "${IMAGE_EMB}" \
        --caption_embeddings "${CAPTION_EMB}"
done

uv run --no-sync python scripts/compute-embeddings/corrupt_teacher_embeddings.py \
    --mode random \
    --seed "${SEED}" \
    --image_embeddings "${IMAGE_EMB}" \
    --caption_embeddings "${CAPTION_EMB}"
