#!/usr/bin/env python

# pyright: reportPossiblyUnboundVariable=false

import argparse
import json
import os
import random

import numpy as np
import torch
from loguru import logger
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image, ImageFile
from pydantic import BaseModel
from tqdm.autonotebook import tqdm

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


def compute_embeddings(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ":" in args.model:
        model_name, pretrained = args.model.split(":")
    else:
        model_name = args.model
        pretrained = None

    logger.info(f"Using model: {model_name} with pretrained weights: {pretrained} via OpenCLIP")
    model, transform = create_model_from_pretrained(  # pyright: ignore
        model_name=model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = get_tokenizer(model_name)
    tokenizer.set_language(args.language)  # pyright: ignore[reportAttributeAccessIssue]

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
        logger.error(f"Image embeddings file {image_save_file_path} already exists. Exiting..")
        return

    caption_save_file_path = os.path.join(
        dataset_dir, os.path.splitext(metadata_json_file)[0] + "_caption_embeddings.pt"
    )
    if os.path.exists(caption_save_file_path):
        logger.error(f"Caption embeddings file {caption_save_file_path} already exists. Exiting..")
        return

    id_to_image_path = {image.id: image.image_path for image in metadata.images}

    samples: list[
        tuple[str, int, int | str, str]
    ] = []  # (image_path, image_id, caption_id, caption)
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
    image_samples: list[str] = []
    caption_counts: list[int] = []
    i = 0
    while i < len(samples):
        j = i
        while j + 1 < len(samples) and samples[j + 1][1] == samples[i][1]:
            j += 1

        image_samples.append(samples[i][0])
        caption_counts.append(j - i + 1)
        i = j + 1

    # construct inputs for the model
    image_inputs = image_samples
    caption_inputs = [caption for _, _, _, caption in samples]

    # sort by caption length in descending order for faster inference
    _permutation = np.argsort([-len(caption) for caption in caption_inputs])
    _inverse_permutation = np.argsort(_permutation)
    caption_inputs = [caption_inputs[idx] for idx in _permutation]

    all_caption_embeddings: list[torch.Tensor] = []
    all_image_embeddings: list[torch.Tensor] = []

    with torch.no_grad():
        logger.info("Computing caption embeddings")

        for i in tqdm(
            range(0, len(caption_inputs), args.batch_size),
            desc="Computing caption embeddings",
        ):
            batch_caption = caption_inputs[i : i + args.batch_size]
            batch_inputs = tokenizer(batch_caption).to(device)
            batch_caption_embeddings = model.encode_text(  # pyright: ignore[reportCallIssue]
                batch_inputs,
                normalize=args.normalize,
            )
            all_caption_embeddings.append(batch_caption_embeddings)

        caption_embeddings = torch.cat(all_caption_embeddings, dim=0).cpu()
        caption_embeddings = caption_embeddings[_inverse_permutation]

        for i in tqdm(
            range(0, len(image_inputs), args.batch_size),
            desc="Computing image embeddings",
        ):
            batch_image_inputs = image_inputs[i : i + args.batch_size]

            batch_image_pils = []
            for image_path in batch_image_inputs:
                try:
                    image = Image.open(image_path)
                    # handle palette images with transparency
                    if image.mode == "P" and "transparency" in image.info:
                        image = image.convert("RGBA")

                    image = image.convert("RGB")

                    batch_image_pils.append(image)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
                    raise e

            batch_image_tensors = torch.stack([transform(img) for img in batch_image_pils]).to(
                device
            )
            batch_image_embeddings = model.encode_image(  # pyright: ignore[reportCallIssue]
                batch_image_tensors,
                normalize=args.normalize,
            )

            del batch_image_pils, batch_image_tensors  # free up memory

            all_image_embeddings.append(batch_image_embeddings)
            del batch_image_embeddings  # free up memory

        image_embeddings = torch.cat(all_image_embeddings, dim=0).cpu()
        # expand image embeddings according to caption counts
        image_embeddings = image_embeddings.repeat_interleave(
            torch.tensor(caption_counts, device=image_embeddings.device), dim=0
        )

    assert caption_embeddings.shape == image_embeddings.shape, (
        f"Caption embeddings shape {caption_embeddings.shape} does not match "
        f"image embeddings shape {image_embeddings.shape}"
    )

    logger.info(f"Caption embeddings shape: {caption_embeddings.shape}")
    logger.info(f"Image embeddings shape: {image_embeddings.shape}")

    torch.save(image_embeddings, image_save_file_path)
    torch.save(caption_embeddings, caption_save_file_path)
    logger.info(f"Saved image embeddings to {image_save_file_path}")
    logger.info(f"Saved caption embeddings to {caption_save_file_path}")


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        choices=["nllb-clip-large-siglip:v1", "nllb-clip-large-siglip:mrl"],
        help="Model to use for computing embeddings (via OpenCLIP)",
        default="nllb-clip-large-siglip:v1",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language to use for tokenization (set via tokenizer.set_language in OpenCLIP)",
        default="vie_Latn",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-Compute caption and image embeddings using nllb-clip- models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    set_seed(42)

    compute_embeddings(args)


if __name__ == "__main__":
    main()
