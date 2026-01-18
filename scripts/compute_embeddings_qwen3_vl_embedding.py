#!/usr/bin/env python

import argparse
import json
import os

import torch
from loguru import logger
from pydantic import BaseModel
from tqdm.autonotebook import tqdm

from qwen3_vl_embedding import Qwen3VLEmbedder


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
        choices=["Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Embedding-8B"],
        help="Pretrained Qwen3-L model to use",
        default="Qwen/Qwen3-VL-Embedding-2B",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type for model weights",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
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
        "--normalize",
        action="store_true",
        help="Whether to normalize the embeddings",
    )


def compute_embeddings(args: argparse.Namespace) -> None:
    model = Qwen3VLEmbedder(
        args.model,
        dtype=args.dtype,
        attn_implementation="flash_attention_2",
        max_pixels=345600,  # 480x720
    )

    dataset_dir = args.dataset_dir
    metadata_json_file = args.metadata_json_file
    metadata_file_path = os.path.join(dataset_dir, metadata_json_file)

    logger.info(f"Loading image text data from: {metadata_file_path}")
    with open(metadata_file_path, "r") as f:
        metadata = ImageTextData.model_validate(json.load(f))

    logger.info(
        f"Found {len(metadata.images)} images and {len(metadata.annotations)} annotations."
    )

    image_save_file_path = os.path.join(
        dataset_dir, os.path.splitext(metadata_json_file)[0] + "_image_embeddings.pt"
    )
    if os.path.exists(image_save_file_path):
        logger.error(f"Caption embeddings file {image_save_file_path} already exists. Exiting..")
        return

    caption_save_file_path = os.path.join(
        dataset_dir, os.path.splitext(metadata_json_file)[0] + "_caption_embeddings.pt"
    )
    if os.path.exists(caption_save_file_path):
        logger.error(f"Caption embeddings file {caption_save_file_path} already exists. Exiting..")
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

    ordered_samples_image = [{"image": image_path} for image_path, _, _, _ in samples]
    ordered_samples_caption = [{"text": caption} for _, _, _, caption in samples]

    with torch.inference_mode():
        for i in tqdm(range(0, len(ordered_samples_image), args.batch_size)):
            batch_image = ordered_samples_image[i : i + args.batch_size]

            batch_image_embeddings = model.process(
                batch_image,
                normalize=args.normalize,
            )
            if i == 0:
                image_embeddings = batch_image_embeddings
            else:
                image_embeddings = torch.cat((image_embeddings, batch_image_embeddings), dim=0)

        for i in tqdm(range(0, len(ordered_samples_caption), args.batch_size)):
            batch_caption = ordered_samples_caption[i : i + args.batch_size]
            batch_caption_embeddings = model.process(
                batch_caption,
                normalize=args.normalize,
            )
            if i == 0:
                caption_embeddings = batch_caption_embeddings
            else:
                caption_embeddings = torch.cat(
                    (caption_embeddings, batch_caption_embeddings), dim=0
                )

    caption_embeddings = caption_embeddings.cpu()
    image_embeddings = image_embeddings.cpu()
    # save to disk to load later for computing OT transport plan during training
    torch.save(image_embeddings, image_save_file_path)
    torch.save(caption_embeddings, caption_save_file_path)

    logger.info(f"Saved image embeddings to {image_save_file_path}")
    logger.info(f"Saved caption embeddings to {caption_save_file_path}")


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
