# Remove near-duplicate images from the dataset

Scripts in this directory are used to identify and remove near-duplicate images from the dataset based on pre-computed embeddings.

These scripts are adapted from [Hugging Face's Image Deduplication Toolkit](https://github.com/huggingface/large-scale-image-deduplication), which is built around Facebook's [SSCD (Self-Supervised Copy Detection)](https://github.com/facebookresearch/sscd-copy-detection) model.

## Download pre-trained SSCD models
```bash
mkdir -p models

# download sscd_imagenet_mixup model
wget -O models/sscd_disc_mixup.torchscript.pt https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
```

## Dataset structure

Your dataset directory should have the following structure:

```
dataset_root/
├── images
│    ├── 000001.jpg
│    ├── 000002.png
│    ├── ...
│    └── nnnnnn.jpg
├── train.json
├── test.json
└── val.json
```

Where `train.json`, `test.json`, and `val.json` are metadata files and have the format:
```json
{
    "images": [
        {"id": 1, "image_path": "images/000001.jpg"},
        {"id": 2, "image_path": "images/000002.png"},
        ...
    ],
    "annotations": [
        {"id": 1, "caption": "A caption for image 1", "image_id": 1},
        {"id": 2, "caption": "A caption for image 2", "image_id": 2},
        ...
    ]
}
```

## Pre-compute embeddings

To pre-compute image embeddings for your dataset, run the following command:

```bash
python scripts/remove_near_duplicates/compute_embeddings.py \
    --dataset_dir /path/to/your/dataset \
    --split_name train \
    --output_dir /path/to/save/embeddings \
    --device auto \
    --batch_size 32
```

## Find near-duplicate images

To compare your dataset against pre-computed embeddings and identify near-duplicate images, run the following command:

```bash
python scripts/remove_near_duplicates/dedup.py \
    --dataset_dir /path/to/your/dataset \
    --split_name test \
    --precomputed_dir /path/to/precomputed/embeddings \
    --precomputed_split_name train \
    --threshold 0.9 \
    --output_dir /path/to/save/duplicates \
    --device auto \
    --batch_size 32
```
