#!/usr/bin/env python

"""Adapted from: https://github.com/huggingface/large-scale-image-deduplication/blob/main/compute_embeddings.py"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from loguru import logger
from PIL import Image, ImageFile
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

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


class ImageTextDataset(Dataset[tuple[Image.Image | Tensor, int]]):
    def __init__(
        self,
        root_dir: str,
        metadata_json_file: str,
        image_transforms=None,
    ) -> None:
        self.root_dir = root_dir
        self.metadata_file_path = os.path.join(self.root_dir, metadata_json_file)
        self.image_transforms = image_transforms

        logger.info(f"Loading image text data from: {self.metadata_file_path}")
        with open(self.metadata_file_path, "r") as f:
            self.metadata = ImageTextData.model_validate(json.load(f))

        logger.info(
            f"Found {len(self.metadata.images)} images and {len(self.metadata.annotations)} annotations."
        )

        image_ids_set = {image.id for image in self.metadata.images}
        image_ids = [(image.id, image.image_path) for image in self.metadata.images]

        # validate image ids
        for annotation in self.metadata.annotations:
            image_id = annotation.image_id
            if image_id not in image_ids_set:
                raise ValueError(
                    f"Annotation with id {annotation.id} has invalid image_id {image_id} "
                    f"not found in images."
                )

        # sort by image id
        sorted_image_ids = sorted(image_ids, key=lambda x: x[0])
        self.images = [
            (image_id, os.path.join(self.root_dir, image_path))
            for image_id, image_path in sorted_image_ids
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> tuple[Image.Image | Tensor, int]:
        image_id, image_path = self.images[idx]

        try:
            image = Image.open(image_path)
            # handle palette images with transparency
            if image.mode == "P" and "transparency" in image.info:
                image = image.convert("RGBA")

            image = image.convert("RGB")

        except (OSError, SyntaxError) as e:
            logger.warning(f"Corrupt image at {image_path}, skipping. Error: {e}")
            # recursively get the next image
            return self.__getitem__((idx + 1) % len(self))

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        return image, image_id


def load_model(model_path: str = "models/sscd_disc_mixup.torchscript.pt", device=None):
    """Load and setup the model."""
    model = torch.jit.load(model_path)
    model.eval()
    if device:
        model = model.to(device)
    return model


def create_transforms():
    """Create and return image transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Resize([320, 320]),
            transforms.ToTensor(),
            normalize,
        ]
    )


def compute_batch_embeddings(model, dataloader, device):
    """Compute embeddings for all batches and return results with timing info."""
    embeddings_list = []
    image_ids = []
    model_inference_time = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            batch_images, batch_indices = batch
            batch_tensor = batch_images.to(device)
            start_model_time = time.time()
            embeddings = model(batch_tensor)
            end_model_time = time.time()
            model_inference_time += end_model_time - start_model_time
            embeddings_list.append(embeddings.cpu().numpy())
            image_ids.extend(batch_indices)

    return embeddings_list, image_ids, model_inference_time


def save_results(embeddings, image_ids: list[int], output_dir: str):
    """Save embeddings and image IDs to files."""
    os.makedirs(output_dir, exist_ok=True)
    save_file_name = os.path.basename(output_dir.rstrip("/"))
    # Include name in filename if provided

    np.save(os.path.join(output_dir, f"{save_file_name}_embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, f"{save_file_name}_image_ids.npy"), np.array(image_ids))


def print_results(embeddings, total_time, model_inference_time, output_dir):
    """Print timing and result statistics."""
    num_embeddings = len(embeddings)
    time_per_sample = total_time / num_embeddings if num_embeddings > 0 else 0
    model_time_per_sample = model_inference_time / num_embeddings if num_embeddings > 0 else 0

    logger.info(f"Saved {num_embeddings} embeddings to {output_dir}/")
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Total time: {total_time:.5f} seconds")
    logger.info(f"Time per sample: {time_per_sample:.5f} seconds")
    logger.info(f"Model inference time: {model_inference_time:.5f} seconds")
    logger.info(f"Model time per sample: {model_time_per_sample:.5f} seconds")


def compute_embeddings(
    dataset_dir: str,
    output_dir: str,
    device: str = "auto",
    batch_size: int = 32,
    split_name: str = "train",
):
    """Compute embeddings for all images in a HuggingFace dataset."""
    function_start_time = time.time()

    # Setup components
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)  # pyright: ignore
    model = load_model(device=device)
    transform = create_transforms()

    # Load dataset and add explicit indices
    dataset = ImageTextDataset(
        root_dir=dataset_dir,
        metadata_json_file=f"{split_name}.json",
        image_transforms=transform,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Compute embeddings
    start_time = time.time()
    embeddings_list, image_ids, model_inference_time = compute_batch_embeddings(
        model, dataloader, device
    )
    end_time = time.time()

    # Process results
    all_embeddings = np.vstack(embeddings_list)
    total_time = end_time - start_time

    # Save and report results
    save_results(all_embeddings, image_ids, output_dir)
    print_results(all_embeddings, total_time, model_inference_time, output_dir)

    function_end_time = time.time()
    function_total_time = function_end_time - function_start_time
    logger.info(f"Total function time: {function_total_time:.5f} seconds")


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for computation",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        help="Dataset split to process",
        default="train",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute embeddings for dataset",
    )
    add_opts(parser)
    args = parser.parse_args()

    compute_embeddings(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()
