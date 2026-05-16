#!/usr/bin/env python

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformers import AutoModel, AutoProcessor

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


@dataclass
class LatencyStats:
    model_family: str
    model_name: str
    modality: str
    num_samples: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    return img.convert("RGB")


class EmbeddingModelWrapper(ABC):
    _SUPPORTED_MODELS = []

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        normalize_embeddings: bool = True,
    ) -> None:
        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model for {self.__class__.__name__} class: {model_name}"
            )

        if isinstance(device, str):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
        else:
            self.device = device
        self.normalize_embeddings = normalize_embeddings

        logger.info(f"Using device: {self.device}")

    @abstractmethod
    def encode_captions(self, captions: list[str], batch_size: int = 32) -> Tensor: ...

    @abstractmethod
    def encode_images(
        self, images: list[str | Path | Image.Image], batch_size: int = 32
    ) -> Tensor: ...

    def encode_one(
        self,
        sample: str | Path | Image.Image,
        modality: Literal["image", "caption"],
        batch_size: int = 32,
    ) -> Tensor:
        if modality == "caption":
            return self.encode_captions([str(sample)], batch_size=batch_size)
        return self.encode_images([sample], batch_size=batch_size)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class SigLIPWrapper(EmbeddingModelWrapper):
    _SUPPORTED_MODELS = [
        "google/siglip-base-patch16-256-multilingual",
    ]

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-256-multilingual",
        device: str | torch.device = "auto",
        dtype: str = "float32",
        use_flash_attn: bool = False,
        max_length: int = 62,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

        model_kwargs: dict[str, Any] = {}
        if use_flash_attn and dtype != "float32":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=dtype,
                **model_kwargs,
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length

    @torch.inference_mode()
    def encode_captions(self, captions: list[str], batch_size: int = 32) -> Tensor:
        embeddings = []
        for i in range(0, len(captions), batch_size):
            batch_inputs = self.processor(
                text=captions[i : i + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            batch_embeddings = self.model.get_text_features(**batch_inputs)
            embeddings.extend(batch_embeddings)

        embeddings = torch.stack(embeddings, dim=0)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @torch.inference_mode()
    def encode_images(
        self, images: list[str | Path | Image.Image], batch_size: int = 32
    ) -> Tensor:
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch_inputs = self.processor(
                images=[load_rgb_image(image) for image in images[i : i + batch_size]],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            batch_embeddings = self.model.get_image_features(**batch_inputs)
            embeddings.extend(batch_embeddings)

        embeddings = torch.stack(embeddings, dim=0)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class NLLBCLipWrapper(EmbeddingModelWrapper):
    _SUPPORTED_MODELS = [
        "nllb-clip-large-siglip:v1",
    ]

    def __init__(
        self,
        model_name: str = "nllb-clip-large-siglip:v1",
        device: str | torch.device = "auto",
        language: str = "vie_Latn",
        max_length: int = 62,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

        from open_clip import create_model_from_pretrained, get_tokenizer

        if ":" in model_name:
            model_name, pretrained = model_name.split(":")
        else:
            model_name = model_name
            pretrained = None

        logger.info(
            f"Using model: {model_name} with pretrained weights: {pretrained} via OpenCLIP"
        )
        self.model, self.transform = create_model_from_pretrained(  # pyright: ignore
            model_name=model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = get_tokenizer(model_name)
        self.tokenizer.set_language(language)  # pyright: ignore[reportAttributeAccessIssue]

        self.max_length = max_length

    @torch.inference_mode()
    def encode_captions(self, captions: list[str], batch_size: int = 32) -> Tensor:
        embeddings = []
        for i in range(0, len(captions), batch_size):
            batch_inputs = self.tokenizer(
                captions[i : i + batch_size],
            ).to(self.device)

            batch_embeddings = self.model.encode_text(  # pyright: ignore[reportCallIssue]
                batch_inputs, normalize=self.normalize_embeddings
            )
            embeddings.extend(batch_embeddings)

        embeddings = torch.stack(embeddings, dim=0)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @torch.inference_mode()
    def encode_images(
        self, images: list[str | Path | Image.Image], batch_size: int = 32
    ) -> Tensor:
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch_image_tensors = torch.stack(
                [self.transform(load_rgb_image(image)) for image in images[i : i + batch_size]]
            ).to(self.device)

            batch_embeddings = self.model.encode_image(  # pyright: ignore[reportCallIssue]
                batch_image_tensors,
                normalize=self.normalize_embeddings,
            )
            embeddings.extend(batch_embeddings)

        embeddings = torch.stack(embeddings, dim=0)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class JinaWrapper(EmbeddingModelWrapper):
    _SUPPORTED_MODELS = ["jinaai/jina-embeddings-v4", "jinaai/jina-clip-v2"]

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str | torch.device = "auto",
        dtype: str = "bfloat16",
        use_flash_attn: bool = False,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

        model_kwargs: dict[str, Any] = {"dtype": dtype}
        if use_flash_attn and dtype != "float32":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = SentenceTransformer(
            model_name,
            device=str(self.device),
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        ).eval()

    @torch.inference_mode()
    def encode_captions(self, captions: list[str], batch_size: int = 32) -> Tensor:
        return self.model.encode(
            captions,
            normalize_embeddings=self.normalize_embeddings,
            task="retrieval",
            convert_to_tensor=True,
            batch_size=batch_size,
        )

    @torch.inference_mode()
    def encode_images(
        self, images: list[str | Path | Image.Image], batch_size: int = 32
    ) -> Tensor:
        return self.model.encode(  # pyright: ignore[reportCallIssue]
            [load_rgb_image(image) for image in images],  # pyright: ignore[reportArgumentType]
            normalize_embeddings=self.normalize_embeddings,
            task="retrieval",
            convert_to_tensor=True,
            batch_size=batch_size,
        )


class Qwen3VLEmbeddingWrapper(EmbeddingModelWrapper):
    _SUPPORTED_MODELS = [
        "Qwen/Qwen3-VL-Embedding-2B",
        "Qwen/Qwen3-VL-Embedding-8B",
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | torch.device = "auto",
        dtype: str = "auto",
        max_pixels: int = 1310720,
        instruction: str = "Retrieve images or text relevant to the user's query.",
        use_flash_attn: bool = False,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

        from qwen3_vl_embedding import Qwen3VLEmbedder

        model_kwargs: dict[str, Any] = {}
        if use_flash_attn and dtype != "float32":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = Qwen3VLEmbedder(
            model_name,
            dtype=dtype,
            max_pixels=max_pixels,
            **model_kwargs,
        )
        self.instruction = instruction

    @torch.inference_mode()
    def encode_captions(self, captions: list[str], batch_size: int = 32) -> Tensor:
        embeddings = []
        for i in range(0, len(captions), batch_size):
            batch_inputs = [
                {
                    "text": caption,
                    "instruction": self.instruction,
                }
                for caption in captions[i : i + batch_size]
            ]

            batch_embeddings = self.model.process(
                inputs=batch_inputs,
                normalize=self.normalize_embeddings,
            )
            embeddings.extend(batch_embeddings)

        embeddings = torch.stack(embeddings, dim=0)

        return embeddings

    @torch.inference_mode()
    def encode_images(
        self, images: list[str | Path | Image.Image], batch_size: int = 32
    ) -> Tensor:
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch_inputs = [
                {
                    "image": load_rgb_image(image),
                    "instruction": self.instruction,
                }
                for image in images[i : i + batch_size]
            ]

            batch_embeddings = self.model.process(
                batch_inputs,
                normalize=self.normalize_embeddings,
            )
            embeddings.extend(batch_embeddings)

        embeddings = torch.stack(embeddings, dim=0)

        return embeddings


class ViCLIPOTWrapper(EmbeddingModelWrapper):
    _SUPPORTED_MODELS = [
        "minhnguyent546/ViCLIP-OT",
        "minhnguyent546/ViSigLIP-OT",
    ]

    def __init__(
        self,
        model_name: str = "minhnguyent546/ViSigLIP-OT",
        device: str | torch.device = "auto",
        dtype: str = "float32",
        use_flash_attn: bool = False,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

        model_kwargs: dict[str, Any] = {"dtype": dtype}
        if use_flash_attn and dtype != "float32":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = SentenceTransformer(
            model_name,
            device=str(self.device),
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        ).eval()

    @torch.inference_mode()
    def encode_captions(self, captions: list[str], batch_size: int = 32) -> Tensor:
        return self.model.encode(
            captions,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=True,
        )

    @torch.inference_mode()
    def encode_images(
        self, images: list[str | Path | Image.Image], batch_size: int = 32
    ) -> Tensor:
        return self.model.encode(  # pyright: ignore[reportCallIssue]
            [load_rgb_image(image) for image in images],  # pyright: ignore[reportArgumentType]
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=True,
        )


def measure_latency(args: argparse.Namespace) -> None:
    logger.info("Environment info:")
    for lib in ("torch", "transformers", "sentence_transformers", "flash-attn"):
        try:
            if lib == "torch":
                # call torch.__version__ to include cuda toolkit version
                lib_version = torch.__version__
            else:
                lib_version = version(lib)
        except Exception:
            lib_version = "N/A"
        logger.info(f"  {lib}: {lib_version}")

    kwargs = {
        "device": args.device,
        "dtype": args.dtype,
        "normalize_embeddings": args.normalize_embeddings,
        "use_flash_attn": args.use_flash_attn,
        "model_name": args.model_name,
    }
    if args.model_family == "qwen3-vl-embedding":
        kwargs["instruction"] = args.instruction
    elif args.model_family == "nllb-clip":
        kwargs["language"] = ("vie_Latn",)

    logger.info(f"Loading wrapper for {args.model_family}...")
    model_wrapper = create_model_wrapper(args.model_family, **kwargs)
    captions, image_paths = load_samples(
        dataset_dir=args.dataset_dir,
        metadata_json_file=args.metadata_json_file,
        max_num_images=args.max_num_images,
        max_num_captions=args.max_num_captions,
    )
    logger.info(f"Measuring {len(captions)} captions and {len(image_paths)} images")
    logger.info(
        f"Number of warmup samples set to {args.warmup_images} for images and {args.warmup_captions} for captions"
    )
    caption_ms = measure_one_by_one(
        model_wrapper=model_wrapper,
        samples=captions,
        modality="caption",
        warmup=args.warmup_captions,
    )
    image_ms = measure_one_by_one(
        model_wrapper=model_wrapper,
        samples=image_paths,
        modality="image",
        warmup=args.warmup_images,
    )

    model_name = args.model_name
    rows = [
        summarize(caption_ms, args.model_family, model_name, "caption"),
        summarize(image_ms, args.model_family, model_name, "image"),
    ]

    file_exists = os.path.exists(args.output_csv)
    with open(args.output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
            logger.info(row)


def create_model_wrapper(family: str, **kwargs: Any) -> EmbeddingModelWrapper:
    family = family.lower()
    if family in {"msiglip"}:
        return SigLIPWrapper(**kwargs)
    if family in {"nllb-clip"}:
        return NLLBCLipWrapper(**kwargs)
    if family in {"jina-clip-v2", "jina-embeddings-v4"}:
        return JinaWrapper(**kwargs)
    if family in {"qwen3-vl-embedding"}:
        return Qwen3VLEmbeddingWrapper(**kwargs)
    if family in {"viclip-ot"}:
        return ViCLIPOTWrapper(**kwargs)
    raise ValueError(f"Unsupported model family: {family}")


def summarize(values: list[float], family: str, model_name: str, modality: str) -> LatencyStats:
    return LatencyStats(
        model_family=family,
        model_name=model_name,
        modality=modality,
        num_samples=len(values),
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p90_ms=np.percentile(values, 90.0),  # pyright: ignore[reportArgumentType]
        p95_ms=np.percentile(values, 95.0),  # pyright: ignore[reportArgumentType]
        p99_ms=np.percentile(values, 99.0),  # pyright: ignore[reportArgumentType]
        min_ms=min(values),
        max_ms=max(values),
        std_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def load_samples(
    dataset_dir: str,
    metadata_json_file: str,
    max_num_images: int | None,
    max_num_captions: int | None = None,
) -> tuple[list[str], list[str]]:
    metadata_path = os.path.join(dataset_dir, metadata_json_file)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = ImageTextData.model_validate(json.load(f))

    id_to_image_path = {img.id: img.image_path for img in metadata.images}
    captions = [
        ann.caption
        for ann in sorted(metadata.annotations, key=lambda item: (item.image_id, item.id))
    ]
    referenced = {ann.image_id for ann in metadata.annotations if ann.image_id in id_to_image_path}
    image_paths = [
        os.path.join(dataset_dir, id_to_image_path[image_id]) for image_id in sorted(referenced)
    ]

    if max_num_images is not None:
        image_paths = image_paths[:max_num_images]
    if max_num_captions is not None:
        captions = captions[:max_num_captions]

    return captions, image_paths


def measure_one_by_one(
    model_wrapper: EmbeddingModelWrapper,
    samples: list[str] | list[Path],
    modality: Literal["image", "caption"],
    warmup: int,
) -> list[float]:
    if not samples:
        raise ValueError(
            f"No {modality} samples available for latency measurement. "
            "Check the dataset metadata and max_num_images/max_num_captions settings."
        )
    for sample in tqdm(samples[:warmup], desc="Warming up"):
        model_wrapper.encode_one(sample, modality)
    model_wrapper.synchronize()

    latencies_ms: list[float] = []
    for sample in tqdm(samples, desc=f"Measuring {modality} latency"):
        model_wrapper.synchronize()
        start = time.perf_counter()
        model_wrapper.encode_one(sample, modality)
        model_wrapper.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms


def add_opts(parser: argparse.ArgumentParser) -> None:
    # model, dataset, and outputs
    parser.add_argument(
        "--model_family",
        required=True,
        type=str,
        choices=[
            "msiglip",
            "nllb-clip",
            "jina-clip-v2",
            "jina-embeddings-v4",
            "qwen3-vl-embedding",
            "viclip-ot",
        ],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="HF model id to use for inference",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the dataset directory",
        default="./data/UIT-OpenViIC",
    )
    parser.add_argument(
        "--metadata_json_file",
        type=str,
        help="Metadata JSON file containing the dataset annotations",
        default="val.json",
    )
    parser.add_argument(
        "--output_csv",
        default="latency_results.csv",
    )

    # model configs, e.g., device, dtype
    parser.add_argument(
        "--device",
        type=str,
        help="Which device to use for inference",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="dtype to use for loading model",
        choices=["float32", "float16", "bfloat16", "auto"],
        default="float32",
    )

    # encoding configs
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size used for inference",
        default=32,
    )
    parser.add_argument(
        "--max_num_images",
        type=int,
        help="Maximum number of images used for inference. Leave None to use all available samples",
        default=None,
    )
    parser.add_argument(
        "--max_num_captions",
        type=int,
        help="Maximum number of captions used for inference. Leave None to use all available samples",
        default=None,
    )
    parser.add_argument(
        "--warmup_images",
        type=int,
        help="Number of images used for warming up",
        default=50,
    )
    parser.add_argument(
        "--warmup_captions",
        type=int,
        help="Number of captions used for warming up",
        default=50,
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="Whether to normalize embeddings",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Enable using `flash_attention_2` for inference",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Instruction to use for encoding if the model support instruction-aware encoding",
        default="Retrieve images or text relevant to the user's query.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_opts(parser)
    args = parser.parse_args()

    measure_latency(args)


if __name__ == "__main__":
    main()
