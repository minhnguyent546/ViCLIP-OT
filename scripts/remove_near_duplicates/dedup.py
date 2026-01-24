#!/usr/bin/env python

import argparse
import json
import os

from loguru import logger
from PIL import ImageFile
from pydantic import BaseModel

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextDataImage(BaseModel):
    id: int
    image_path: str


class ImageTextDataAnnotation(BaseModel):
    id: int
    caption: str
    image_id: int


class ImageTextData(BaseModel):
    images: list[ImageTextDataImage]
    annotations: list[ImageTextDataAnnotation]


def remove_duplicate_images(args: argparse.Namespace) -> None:
    train_metadata_json_file = os.path.join(args.dataset_dir, f"{args.split_name}.json")

    with open(train_metadata_json_file, "r") as f:
        metadata = ImageTextData.model_validate(json.load(f))

    orig_num_images = len(metadata.images)
    orig_num_annotations = len(metadata.annotations)

    duplicate_result_file = os.path.join(
        args.duplicate_result_dir, f"{args.split_name}_duplicates.json"
    )
    if not os.path.isfile(duplicate_result_file):
        raise FileNotFoundError(f"Duplicate result file not found: {duplicate_result_file}")

    with open(duplicate_result_file, "r") as f:
        duplicate_results = json.load(f)

    if "duplicate_image_ids" not in duplicate_results:
        raise ValueError(
            "Invalid duplicate result file format: missing 'duplicate_image_ids' field."
        )
    duplicate_image_ids = set(duplicate_results["duplicate_image_ids"])
    new_images = []
    new_annotations = []
    removed_image_count = 0
    for image in metadata.images:
        if image.id in duplicate_image_ids:
            removed_image_count += 1
        else:
            new_images.append(image)

    remove_caption_count = 0
    for annotation in metadata.annotations:
        if annotation.image_id in duplicate_image_ids:
            remove_caption_count += 1
        else:
            new_annotations.append(annotation)

    remove_image_ratio = removed_image_count / orig_num_images
    remove_caption_ratio = remove_caption_count / orig_num_annotations
    logger.info(
        f"Removed {removed_image_count}/{orig_num_images} ({remove_image_ratio:.2%}) duplicate images and {remove_caption_count}/{orig_num_annotations} ({remove_caption_ratio:.2%}) captions from {args.split_name} split metadata file."
    )

    save_file_path = os.path.join(args.dataset_dir, f"{args.split_name}_dedup.json")
    new_metadata = ImageTextData(images=new_images, annotations=new_annotations)
    with open(save_file_path, "w") as f:
        json.dump(new_metadata.model_dump(), f, ensure_ascii=False)

    logger.info(f"Saved deduplicated metadata to {save_file_path}.")


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing metadata JSON files.",
    )
    parser.add_argument(
        "--duplicate_result_dir",
        type=str,
        required=True,
        help="Path to the directory containing duplicate result JSON files.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        help="Dataset split to process",
        default="train",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate images from dataset based on precomputed duplicate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    remove_duplicate_images(args)


if __name__ == "__main__":
    main()
