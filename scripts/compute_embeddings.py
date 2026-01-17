#!/usr/bin/env python

import argparse
import json
import os

import torch
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


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


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        help="Pretrained sentence-transformer model to use",
        default="google/embeddinggemma-300m",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        help="Prompt name for the sentence-transformer model (use should read model usage first to use correct prompt)",
        default="STS",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory containing the dataset",
        default="./data/UIT-OpenViIC",
    )
    parser.add_argument(
        "--metadata_json_file",
        type=str,
        help="Metadata JSON file containing the dataset annotations",
        default="train.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for processing captions",
        default=32,
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=bool,
        help="Whether to normalize embeddings",
        default=True,
    )


def compute_embeddings(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model, device=device.type)
    model.eval()

    dataset_dir = args.dataset_dir
    metadata_json_file = args.metadata_json_file
    metadata_file_path = os.path.join(dataset_dir, metadata_json_file)

    logger.info(f"Loading image text data from: {metadata_file_path}")
    with open(metadata_file_path, "r") as f:
        metadata = ImageTextData.model_validate(json.load(f))

    logger.info(
        f"Found {len(metadata.images)} images and {len(metadata.annotations)} annotations."
    )
    save_file_path = os.path.join(
        dataset_dir, os.path.splitext(metadata_json_file)[0] + "_caption_embeddings.pt"
    )
    if os.path.exists(save_file_path):
        logger.error(f"Caption embeddings file {save_file_path} already exists. Exiting..")
        return

    id_to_image_path = {image.id: image.image_path for image in metadata.images}

    samples: list[tuple[str, int, int, str]] = []
    for annotation in metadata.annotations:
        image_id = annotation.image_id
        if image_id not in id_to_image_path:
            raise RuntimeError(
                f"Could not find image with ID {image_id} for annotation {annotation.id}"
            )

        image_path = os.path.join(dataset_dir, id_to_image_path[image_id])
        samples.append((image_path, image_id, annotation.id, annotation.caption))

    # sorted by image_id, annotation_id
    samples.sort(key=lambda x: (x[1], x[2]))

    ordered_captions = [captions for _, _, _, captions in samples]
    caption_embeddings = []
    with torch.inference_mode():
        caption_embeddings = model.encode(
            ordered_captions,
            prompt_name=args.prompt_name,
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize_embeddings,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    caption_embeddings = caption_embeddings.cpu()
    # save to disk to load later for computing OT transport plan during training
    torch.save(caption_embeddings, save_file_path)
    logger.info(f"Saved caption embeddings to {save_file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute caption embeddings using a sentence-transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    compute_embeddings(args)


if __name__ == "__main__":
    main()
