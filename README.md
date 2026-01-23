# ViCLIP-OT &mdash; The First Foundation Vision-Language Model for Vietnamese Image–Text Retrieval with Optimal Transport

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue?logo=python">
  <a href="https://github.com/minhnguyent546/ViCLIP-OT/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/minhnguyent546/ViCLIP-OT"></a>
  <a href="https://github.com/minhnguyent546/ViCLIP-OT/issues"><img alt="Issues" src="https://img.shields.io/github/issues/minhnguyent546/ViCLIP-OT"></a>
  <a href="https://github.com/minhnguyent546/ViCLIP-OT/pulls"><img alt="PRs" src="https://img.shields.io/github/issues-pr/minhnguyent546/ViCLIP-OT"></a>
</p>

<p align="center">
  <img src="./assets/ViCLIP_OT.jpg" alt="ViCLIP-OT" width="768px">
</p>


> **Abstract:** WIP

---

Table of Contents
=================

- [ViCLIP-OT — The First Foundation Vision-Language Model for Vietnamese Image–Text Retrieval with Optimal Transport](#viclip-ot--the-first-foundation-vision-language-model-for-vietnamese-imagetext-retrieval-with-optimal-transport)
- [Table of Contents](#table-of-contents)
  - [1. Installation](#1-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [2. Datasets](#2-datasets)
    - [Dataset structure](#dataset-structure)
  - [3. Training](#3-training)
  - [4. Inference](#4-inference)
  - [5. Citing](#5-citing)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## 1. Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) - A Package and Project manager for Python

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/minhnguyent546/viclip_ot.git
   cd viclip_ot
   ```

2. **Set up Python environment using uv:**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync

   # Activate virtual environment
   source .venv/bin/activate
   ```

3. **Verify installation:**
   ```bash
   python -m viclip_ot.train --help
   ```

## 2. Datasets

> The dataset used in this study is not publicly available due to institutional or licensing restrictions. However, it can be made available for academic use upon reasonable request. Interested researchers may contact the authors for further information.

### Dataset structure

```
WIP
```

## 3. Training

To train the model, you can run the following command:
```bash
uv run python -m viclip_ot.train \
  --seed 42 \
  --model_config ./config/model.vit_base_dinov3.yaml \
  --dataset_dir ./data/UIT-OpenViIC \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --train_crop_size 224 \
  --eval_resize_size 256 \
  --eval_crop_size 224 \
  --checkpoints_dir ./checkpoints \
  --num_epochs 30 \
  --num_workers 8 \
  --log_file_interval 3 \
  --mixed_precision bf16 \
  --gradient_accum_steps 8 \
  --lr 2e-4 \
  --backbone_lr 5e-5 \
  --lock_image \
  --lock_image_last_unfreeze_groups 2 \
  --weight_decay 1e-4 \
  --scheduler one_cycle_lr \
  --min_lr 1e-6 \
  --lr_warmup_epochs 2 \
  --lr_warmup_method linear \
  --best_checkpoint_metrics t2i_R__1 i2t_R__1 \
  --save_best_k 5 \
  --max_grad_norm 1.0 \
  --wandb_logging \
  --wandb_project viclip_ot_test \
  --wandb_name cliploss_vit_base_dinov3_gemma300m_openviic_final_30
```

## 4. Inference

To run inference on a trained model, you can use the following command:
```bash
uv run python -m viclip_ot.train \
  --run_test_only \
  --from_checkpoint <PATH_TO_YOUR_CHECKPOINT> \
  --seed 42 \
  --model_config ./config/model.vit_base_dinov3.yaml \
  --dataset_dir ./data/UIT-OpenViIC \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --train_crop_size 224 \
  --eval_resize_size 256 \
  --eval_crop_size 224 \
  --num_workers 4 \
```

## 5. Citing

If you find this repository useful for your research, please consider citing:

```bibtex
WIP
```
