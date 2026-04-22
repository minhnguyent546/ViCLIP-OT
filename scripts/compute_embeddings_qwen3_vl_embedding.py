#!/usr/bin/env python

# `qwen3_vl_embedding` script can be downloaded from: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/2a50926d213628c727f38025982a76f655673f54/scripts/qwen3_vl_embedding.py

import argparse
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageFile
from pydantic import BaseModel
from qwen3_vl_embedding import Qwen3VLEmbedder
from tqdm.autonotebook import tqdm

try:
    from vllm import LLM
    from vllm.multimodal.utils import fetch_image

    _has_vllm = True
except ImportError:
    _has_vllm = False

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
    if args.backend == "vllm" and not _has_vllm:
        logger.error("vLLM is not installed. Please install vLLM to use the vLLM backend.")
        return
    if args.backend == "transformers":
        model_kwargs = {}
        if args.use_flash_attn and args.dtype != "float32":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        model = Qwen3VLEmbedder(
            args.model,
            dtype=args.dtype,
            max_pixels=args.max_pixels,
            **model_kwargs,
        )
    else:
        model = LLM(
            model=args.model,
            runner="pooling",
            dtype=args.dtype,
            trust_remote_code=True,
            max_model_len=args.vllm_max_model_len,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            disable_log_stats=False,
            limit_mm_per_prompt={"video": 0, "image": 1},
            mm_processor_kwargs={
                "max_pixels": args.max_pixels,
            },
            mm_encoder_tp_mode="data",
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

    # construct inputs for the model
    image_inputs = [
        {
            "image": image_path,
            "instruction": args.instruction,
        }
        for image_path in image_samples
    ]
    caption_inputs = [
        {
            "text": caption,
            "instruction": args.instruction,
        }
        for _, _, _, caption in samples
    ]

    image_embeddings = None
    caption_embeddings = None
    inference_start_time = time.perf_counter()
    if args.backend == "transformers":
        _permutation = np.argsort([-len(caption["text"]) for caption in caption_inputs])
        _inverse_permutation = np.argsort(_permutation)
        caption_inputs = [caption_inputs[idx] for idx in _permutation]

        with torch.inference_mode():
            for i in tqdm(
                range(0, len(caption_inputs), args.batch_size),
                desc="Computing caption embeddings",
            ):
                batch_caption = caption_inputs[i : i + args.batch_size]
                batch_caption_embeddings = model.process(  # pyright: ignore
                    batch_caption,
                    normalize=args.normalize,
                )
                if i == 0:
                    caption_embeddings = batch_caption_embeddings
                else:
                    caption_embeddings = torch.cat(
                        (caption_embeddings, batch_caption_embeddings),  # pyright: ignore
                        dim=0,
                    )
            assert caption_embeddings is not None
            caption_embeddings = caption_embeddings.cpu()  # pyright: ignore
            caption_embeddings = torch.stack(
                [caption_embeddings[idx] for idx in _inverse_permutation]
            )

            for i in tqdm(
                range(0, len(image_inputs), args.batch_size),
                desc="Computing image embeddings",
            ):
                batch_image = image_inputs[i : i + args.batch_size]

                batch_image_pils = []
                for sample in batch_image:
                    image_path = sample.pop("image")

                    try:
                        image = Image.open(image_path)
                        # handle palette images with transparency
                        if image.mode == "P" and "transparency" in image.info:
                            image = image.convert("RGBA")

                        image = image.convert("RGB")

                        batch_image_pil = {"image": image, **sample}
                        batch_image_pils.append(batch_image_pil)
                    except Exception as e:
                        logger.error(f"Error loading image {image_path}: {e}")
                        raise e

                batch_image_embeddings = model.process(  # pyright: ignore
                    batch_image_pils,
                    normalize=args.normalize,
                )

                del batch_image_pils  # free up memory

                if i == 0:
                    image_embeddings = batch_image_embeddings
                else:
                    image_embeddings = torch.cat(
                        (image_embeddings, batch_image_embeddings),  # pyright: ignore
                        dim=0,
                    )

                del batch_image_embeddings  # free up memory

            image_embeddings = image_embeddings.cpu()  # pyright: ignore
            # expand image embeddings according to caption counts
            image_embeddings = image_embeddings.repeat_interleave(
                torch.tensor(caption_counts, device=image_embeddings.device), dim=0
            )
    else:
        vllm_caption_inputs = [
            prepare_vllm_inputs(caption_input, model) for caption_input in caption_inputs
        ]
        vllm_image_inputs = [
            prepare_vllm_inputs(image_input, model) for image_input in image_inputs
        ]

        # embed captions
        caption_outputs = model.embed(vllm_caption_inputs)  # pyright: ignore
        caption_embeddings = torch.tensor(
            [output.outputs.embedding for output in caption_outputs]  # type: ignore
        )

        # embed images
        image_outputs = model.embed(vllm_image_inputs)  # pyright: ignore
        image_embeddings = torch.tensor(
            [output.outputs.embedding for output in image_outputs]  # type: ignore
        )
        image_embeddings = image_embeddings.repeat_interleave(
            torch.tensor(caption_counts, device=image_embeddings.device), dim=0
        )

    inference_time = time.perf_counter() - inference_start_time

    assert caption_embeddings is not None and image_embeddings is not None
    assert caption_embeddings.shape == image_embeddings.shape, (
        f"Caption embeddings shape {caption_embeddings.shape} does not match "
        f"image embeddings shape {image_embeddings.shape}"
    )

    logger.info(f"Inference time: {to_hms(inference_time)}")
    logger.info(f"Caption embeddings shape: {caption_embeddings.shape}")
    logger.info(f"Image embeddings shape: {image_embeddings.shape}")

    torch.save(image_embeddings, image_save_file_path)
    torch.save(caption_embeddings, caption_save_file_path)

    logger.info(f"Saved image embeddings to {image_save_file_path}")
    logger.info(f"Saved caption embeddings to {caption_save_file_path}")


def _format_input_to_conversation(
    input_dict: dict[str, Any], default_instruction: str = "Represent the user's input."
) -> list[dict[str, Any]]:
    content = []

    instruction = input_dict.get("instruction") or default_instruction
    text = input_dict.get("text")
    image = input_dict.get("image")

    if image:
        image_content = None
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = "file://" + abs_image_path
        else:
            image_content = image

        if image_content:
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                }
            )

    if text:
        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": ""})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]

    return conversation


def prepare_vllm_inputs(
    input_dict: dict[str, Any],
    llm,
) -> dict[str, Any]:
    conversation = _format_input_to_conversation(input_dict)

    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    multi_modal_data = None
    image = input_dict.get("image")
    if image:
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                try:
                    image_obj = fetch_image(image)
                    multi_modal_data = {"image": image_obj}
                except Exception as e:
                    logger.warning(f"Failed to fetch image {image}: {e}")
            else:
                abs_image_path = os.path.abspath(image)
                if os.path.exists(abs_image_path):
                    image_obj = Image.open(abs_image_path)
                    # handle palette images with transparency
                    if image_obj.mode == "P" and "transparency" in image_obj.info:
                        image_obj = image_obj.convert("RGBA")

                    image_obj = image_obj.convert("RGB")
                    multi_modal_data = {"image": image_obj}
                else:
                    logger.warning(f"Image file not found: {abs_image_path}")
        else:
            multi_modal_data = {"image": image}

    result = {"prompt": prompt_text, "multi_modal_data": multi_modal_data}

    return result


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducibility",
        default=42,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Embedding-8B"],
        help="Pretrained Qwen3-L model to use",
        default="Qwen/Qwen3-VL-Embedding-2B",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        help="Maximum number of pixels for the model (default to 1024x1280=1310720, which is the default max resolution for Qwen3-VL-Embedding models)",
        default=1310720,  # 1024x1280
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Instruction for the model",
        default="Retrieve images or text relevant to the user's query.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for inference",
        choices=["vllm", "transformers"],
        default="vllm",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type for model weights",
        choices=["float32", "float16", "bfloat16", "auto"],
        default="auto",
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
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        help="Maximum length of the model input (prompt + generation). Only applicable if backend is vLLM.",
        default=8192,  # Qwen3-VL-Embedding default context length
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        help="GPU memory utilization for vLLM. Only applicable if backend is vLLM.",
        default=0.9,
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        help="Tensor parallel size for vLLM. Only applicable if backend is vLLM.",
        default=1,
    )


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
