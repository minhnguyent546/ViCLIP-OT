from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from hf.processing_viclip_ot import DEFAULT_QWEN3_INSTRUCTION, format_text_with_instruction


def run_instruction_smoke_tests() -> None:
    query = "example query"
    expected = {
        "google/embeddinggemma-300m": f"sentence similarity | query: {query}",
        "intfloat/multilingual-e5-base": f"query: {query}",
        "qwen/qwen3-embedding-0.6b": f"Instruct: {DEFAULT_QWEN3_INSTRUCTION}\nQuery:{query}",
        "baai/bge-m3": query,
        "keepitreal/vietnamese-sbert": query,
    }
    for model_name, expected_text in expected.items():
        actual_text = format_text_with_instruction(
            query,
            model_name=model_name,
            instruction_mode="auto",
        )
        if actual_text != expected_text:
            raise RuntimeError(
                "Instruction smoke test failed for "
                f"{model_name}: expected={expected_text!r}, got={actual_text!r}"
            )


def validate_from_pretrained(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    run_instruction_smoke_tests()

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(device)
    model.eval()

    texts = ["a cat on a chair", "a street at night"][: args.batch_size]
    images = [
        Image.fromarray(
            np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)
        )
        for _ in range(args.batch_size)
    ]

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    print(f"{outputs.keys() = }")
    required_keys = (
        "text_features",
        "image_features",
        "logits_per_text",
        "logits_per_image",
        "logit_scale",
    )
    for key in required_keys:
        if getattr(outputs, key, None) is None:
            raise RuntimeError(f"Missing output key: {key}")

    print("Validation passed.")
    print(f"image_features shape: {tuple(outputs.image_features.shape)}")
    print(f"text_features shape: {tuple(outputs.text_features.shape)}")
    print(f"logit_scale shape: {tuple(outputs.logit_scale.shape)}")
    if getattr(outputs, "logit_bias", None) is not None:
        print(f"logit_bias shape: {tuple(outputs.logit_bias.shape)}")
    print("AutoProcessor loading: OK")
    print("Instruction smoke tests: OK")


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local path or Hub repo id for from_pretrained.",
    )
    parser.add_argument(
        "--device",
        help="Torch device string, e.g. cpu or cuda.",
        default="cpu",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for smoke-test forward pass.",
        default=2,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Synthetic image size for smoke test.",
        default=224,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ViCLIP-OT HF export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    validate_from_pretrained(args)


if __name__ == "__main__":
    main()
