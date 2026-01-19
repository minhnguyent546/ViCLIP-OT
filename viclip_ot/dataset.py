import json
import os
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset

from viclip_ot.utils.logger import logger


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


class ImageTextDataset(Dataset[tuple[Image.Image | Tensor, list[str], int, int]]):
    """
    Dataset structure:

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
    """

    def __init__(
        self,
        root_dir: str,
        metadata_json_file: str,
        image_transforms=None,
        model_fmt: Literal["gemma", "e5", "qwen3", "bge", "sbert"] = "gemma",
    ) -> None:
        self.root_dir = root_dir
        self.metadata_file_path = os.path.join(self.root_dir, metadata_json_file)
        self.image_transforms = image_transforms
        self.model_fmt = model_fmt

        logger.info(f"Loading image text data from: {self.metadata_file_path}")
        with open(self.metadata_file_path, "r") as f:
            self.metadata = ImageTextData.model_validate(json.load(f))

        logger.info(
            f"Found {len(self.metadata.images)} images and {len(self.metadata.annotations)} annotations."
        )
        self.id_to_image_path = {image.id: image.image_path for image in self.metadata.images}

        captions_by_image_id: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for annotation in self.metadata.annotations:
            image_id = annotation.image_id
            if image_id not in self.id_to_image_path:
                raise RuntimeError(
                    f"Could not find image with ID {image_id} for annotation {annotation.id}"
                )

            captions_by_image_id[image_id].append((annotation.id, annotation.caption))

        self.caption_ids_list: list[list[int]] = []
        self.samples: list[tuple[int, str, list[str]]] = []
        pair_count = 0
        for image_id in sorted(captions_by_image_id.keys()):
            captions_by_image_id[image_id].sort(key=lambda x: x[0])  # sort by caption_id
            captions = [caption for _caption_id, caption in captions_by_image_id[image_id]]
            image_path = os.path.join(self.root_dir, self.id_to_image_path[image_id])
            self.samples.append((image_id, image_path, captions))

            self.caption_ids_list.append(list(range(pair_count, pair_count + len(captions))))
            pair_count += len(captions)

    def get_caption_ids(self, indices: list[int]) -> list[int]:
        caption_ids: list[int] = []
        for idx in indices:
            caption_ids.extend(self.caption_ids_list[idx])
        return caption_ids

    def get_image_ids(self, indices: list[int]) -> list[int]:
        image_ids: list[int] = []
        for idx in indices:
            image_id, _, _ = self.samples[idx]
            image_ids.extend([image_id] * len(self.caption_ids_list[idx]))
        return image_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Image.Image | Tensor, list[str], int, int]:
        image_id, image_path, captions = self.samples[idx]

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

        formatted_captions = None
        if self.model_fmt == "gemma":
            # https://huggingface.co/google/embeddinggemma-300m#prompt-instructions
            # TODO: this prompt is for encode document, consider supporting encode for query.
            formatted_captions = [
                f"sentence similarity | query: {caption}" for caption in captions
            ]

        elif self.model_fmt == "e5":
            # https://huggingface.co/intfloat/multilingual-e5-base#usage
            formatted_captions = [f"query: {caption}" for caption in captions]

        elif self.model_fmt == "qwen3":

            def get_detailed_instruct(task_description: str, query: str) -> str:
                return f"Instruct: {task_description}\nQuery:{query}"

            task = "Given a web search query, retrieve relevant passages that answer the query"

            # https://huggingface.co/qwen/qwen3-embedding-0.6b#usage
            formatted_captions = [
                f"{get_detailed_instruct(task, caption)}" for caption in captions
            ]

        elif self.model_fmt == "bge":
            # https://huggingface.co/baai/bge-m3#usage
            # BGE does not require special formatting
            formatted_captions = [f"{caption}" for caption in captions]

        elif self.model_fmt == "sbert":
            # https://www.sbert.net/docs/usage/semantic_textual_similarity.html
            # SBERT does not require special formatting
            formatted_captions = [f"{caption}" for caption in captions]

        else:
            raise ValueError(
                f"Invalid model_fmt: {self.model_fmt}. "
                f"Expected one of ['gemma', 'e5', 'qwen3', 'bge', 'sbert']"
            )

        # Print first caption for debugging
        # logger.debug(f"Image ID {image_id} first caption: {formatted_captions[0]}")

        return image, formatted_captions, int(image_id), idx


class ImageTextCollate:
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        caption_to_use: Literal["first", "random", "all"] = "all",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_to_use = caption_to_use

    def __call__(self, batch):
        images, captions, image_ids, indices = zip(*batch, strict=True)

        images = torch.stack(images)
        image_ids = torch.tensor(image_ids, dtype=torch.int64)

        if self.caption_to_use == "first":
            captions = [caption[0] for caption in captions]
        elif self.caption_to_use == "random":
            captions = [caption[np.random.randint(0, len(caption) - 1)] for caption in captions]
        elif self.caption_to_use == "all":
            # flatten caption and repeat images accordingly
            flat_captions = []
            repeat_counts = []
            for caption_list in captions:
                flat_captions.extend(caption_list)
                repeat_counts.append(len(caption_list))

            # Repeat images to match flattened captions
            images = torch.repeat_interleave(images, torch.tensor(repeat_counts), dim=0)
            image_ids = torch.repeat_interleave(image_ids, torch.tensor(repeat_counts), dim=0)
            captions = flat_captions
        else:
            raise ValueError(
                f"Invalid caption_to_use: {self.caption_to_use}. "
                f"Expected one of ['first', 'random', 'all']"
            )

        text_inputs = self.tokenizer(
            list(captions),
            padding=True,  # Dynamic padding (pad to longest in batch)
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": images,
            "text_inputs": text_inputs,
            "image_ids": image_ids,
            "indices": indices,
        }
