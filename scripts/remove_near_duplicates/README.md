# Remove near-duplicate images from the dataset

Scripts in this directory are used to identify and remove near-duplicate images from the dataset based on pre-computed embeddings.

These scripts are adapted from [Hugging Face's Image Deduplication Toolkit](https://github.com/huggingface/large-scale-image-deduplication), which is built around Facebook's [SSCD (Self-Supervised Copy Detection)](https://github.com/facebookresearch/sscd-copy-detection) model.

## Download pre-trained SSCD models

For more models, see [facebookresearch/sscd-copy-detection](https://github.com/facebookresearch/sscd-copy-detection).

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
    ],
    "annotations": [
        {"id": 42, "caption": "A caption for image 1", "image_id": 1},
        {"id": 43, "caption": "Another caption for image 1", "image_id": 1},
        {"id": 67, "caption": "A caption for image 2", "image_id": 2},
    ]
}
```

The next sections demonstrate how to use the scripts to deduplicate images for KTVIC test split against UIT-OpenViIC training split.

## Pre-compute embeddings

```bash
python scripts/remove_near_duplicates/compute_embeddings.py \
    --dataset_dir ./data/UIT-OpenViIC \
    --split_name train \
    --output_dir ./data/UIT-OpenViIC-sscd-embeddings \
    --device auto \
    --batch_size 64
```

## Find near-duplicate images

```bash
python scripts/remove_near_duplicates/find_duplicate_images.py \
    --dataset_dir ./data/KTVIC \
    --split_name test \
    --precomputed_dir ./data/UIT-OpenViIC-sscd-embeddings \
    --precomputed_split_name train \
    --threshold 0.8 \
    --output_dir ./data/KTVIC \
    --device auto \
    --batch_size 64
```

## Remove near-duplicate images

```bash
python scripts/remove_near_duplicates/dedup.py \
    --dataset_dir ./data/KTVIC \
    --duplicate_result_dir ./data/KTVIC \
    --split_name test
```
