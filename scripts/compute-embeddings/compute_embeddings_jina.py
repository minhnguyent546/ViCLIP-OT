#!/usr/bin/env python

import argparse
import json
import os
import random

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
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


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        choices=["jinaai/jina-embeddings-v4", "jinaai/jina-clip-v2"],
        help="Jina embeddings model to use",
        default="jinaai/jina-embeddings-v4",
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
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Whether to use flash attention if supported by the model",
    )


def compute_embeddings(args: argparse.Namespace) -> None:
    model_kwargs = {
        "dtype": args.dtype,  # make sure the model support this dtype!
    }
    if args.use_flash_attn and args.dtype != "float32":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = SentenceTransformer(
        args.model,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model {args.model} with {num_params:,} parameters.")

    if hasattr(model, "_first_module"):
        model_first_module = model._first_module()  # pyright: ignore[reportPrivateUsage]
        if hasattr(model_first_module, "processor"):
            processor = model_first_module.processor
            logger.info(f"Processor: {processor}")
            # Set max image pixels (e.g., 512*512 = 262144)
            processor.max_pixels = 345600  # 480 * 720
            logger.info(f"Set processor.max_pixels to {processor.max_pixels}")

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

    ordered_samples_caption = [caption for _, _, _, caption in samples]
    _permutation = np.argsort([-len(caption) for caption in ordered_samples_caption])
    _inverse_permutation = np.argsort(_permutation)
    ordered_samples_caption = [ordered_samples_caption[idx] for idx in _permutation]

    all_caption_embeddings: list[torch.Tensor] = []
    all_image_embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in tqdm(
            range(0, len(ordered_samples_caption), args.batch_size),
            desc="Computing caption embeddings",
        ):
            batch_caption = ordered_samples_caption[i : i + args.batch_size]
            batch_caption_embeddings = model.encode(
                sentences=batch_caption,
                normalize_embeddings=args.normalize,
                task="retrieval",
                convert_to_tensor=True,
            )
            all_caption_embeddings.append(batch_caption_embeddings)

        caption_embeddings = torch.cat(all_caption_embeddings, dim=0)
        caption_embeddings = caption_embeddings[_inverse_permutation]

        for i in tqdm(
            range(0, len(image_samples), args.batch_size),
            desc="Computing image embeddings",
        ):
            batch_image = image_samples[i : i + args.batch_size]

            batch_image_pils = []
            for image_path in batch_image:
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

            batch_image_embeddings = model.encode(
                batch_image_pils,
                normalize_embeddings=args.normalize,
                task="retrieval",
                convert_to_tensor=True,
            )

            del batch_image_pils  # free up memory

            all_image_embeddings.append(batch_image_embeddings)
            del batch_image_embeddings  # free up memory

        image_embeddings = torch.cat(all_image_embeddings, dim=0)
        # expand image embeddings according to caption counts
        image_embeddings = image_embeddings.repeat_interleave(
            torch.tensor(caption_counts, device=image_embeddings.device), dim=0
        )

    assert caption_embeddings.shape == image_embeddings.shape, (
        f"Caption embeddings shape {caption_embeddings.shape} does not match "
        f"image embeddings shape {image_embeddings.shape}"
    )

    caption_embeddings = caption_embeddings.cpu()
    image_embeddings = image_embeddings.cpu()
    # save to disk to load later for computing OT transport plan during training
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute caption embeddings using a sentence-transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    set_seed(42)

    compute_embeddings(args)


if __name__ == "__main__":
    main()
