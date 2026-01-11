import json

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


class ImageTextDataset(Dataset[tuple[Image.Image | Tensor, str]]):
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
        data_file_path: str,
        image_transforms=None,
    ) -> None:
        self.data_file_path = data_file_path
        self.image_transforms = image_transforms

        logger.info(f"Loading image text data from: {self.data_file_path}")
        with open(self.data_file_path, "r") as f:
            raw_data = json.load(f)
            self.data = ImageTextData.model_validate(raw_data)

        logger.info(
            f"Found {len(self.data.images)} images and {len(self.data.annotations)} annotations."
        )
        self.id_to_image_path = {image.id: image.image_path for image in self.data.images}

        # flatten Samples: create a list of (image_path, caption)
        self.samples = []
        for annotation in self.data.annotations:
            image_id = annotation.image_id
            if image_id in self.id_to_image_path:
                self.samples.append((self.id_to_image_path[image_id], annotation.caption))
            else:
                logger.warning(
                    f"Could not find image with ID {image_id} for annotation {annotation.id}"
                )
        logger.info(f"Total (image, text) pairs: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Image.Image | Tensor, str]:
        image_path, caption = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        # https://huggingface.co/google/embeddinggemma-300m#prompt-instructions
        # TODO: this prompt is for encode document, consider supporting encode for query.
        formatted_caption = f"title: none | text: {caption}"

        return image, formatted_caption


class ImageTextCollate:
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images, texts = zip(*batch, strict=True)

        if torch.is_tensor(images[0]):
            images = torch.stack(images)
        else:
            images = torch.stack(images)

        text_inputs = self.tokenizer(
            list(texts),
            padding=True,  # Dynamic padding (pad to longest in batch)
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": images,
            "text_inputs": text_inputs,
        }
