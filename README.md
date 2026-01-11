# ViCLIP-OT

**Abstract:** WIP

---

Table of Contents
=================

- [ViCLIP-OT](#viclip-ot)
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
  --model_config ./config/model.yaml \
  --dataset_dir ./data/UIT-ViIC \
  --train_batch_size 64 \
  --train_crop_size 224 \
  --eval_batch_size 64 \
  --eval_resize_size 256 \
  --eval_crop_size 224 \
  --checkpoints_dir ./checkpoints \
  --num_epochs 25 \
  --num_workers 6 \
  --log_file_interval 5 \
  --mixed_precision fp16 \
  --gradient_accum_steps 2 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --scheduler one_cycle_lr \
  --min_lr 1e-7 \
  --lr_warmup_epochs 4 \
  --lr_warmup_method linear \
  --best_checkpoint_metrics loss \
  --save_best_k 4 \
  --max_grad_norm 1.0 \
  --wandb_logging \
  --wandb_project viclip_ot_test \
  --wandb_name test_01
```

## 4. Inference

To run inference on a trained model, you can use the following command:
```bash
WIP
```

## 5. Citing

If you find this repository useful for your research, please consider citing:

```bibtex
WIP
```
