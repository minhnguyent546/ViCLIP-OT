from dataclasses import dataclass
from typing import Any, Literal

import viclip_ot.constants as C
from viclip_ot.model import ViCLIPOT, ViCLIPOTConfig
from viclip_ot.utils import load_yaml_file

CaptionFormat = Literal["gemma", "e5", "qwen3", "bge", "sbert", "plain"]


@dataclass(frozen=True)
class TrainingModelBundle:
    model: Any
    config: ViCLIPOTConfig | JinaCLIPV2Config
    tokenizer: Any
    max_length: int
    caption_format: CaptionFormat
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    required_image_size: int | None
    backbone_prefixes: tuple[str, ...]
    adapter_prefixes: tuple[str, ...]
    supports_tower_locking: bool
    save_trainable_state_only: bool
    test_only_uses_eval_criterion: bool


def _get_viclip_caption_format(model_name: str) -> CaptionFormat:
    normalized_model_name = model_name.lower()
    if "gemma" in normalized_model_name:
        return "gemma"
    if "e5" in normalized_model_name:
        return "e5"
    if "qwen" in normalized_model_name:
        return "qwen3"
    if "bge" in normalized_model_name:
        return "bge"
    if "sbert" in normalized_model_name:
        return "sbert"
    raise ValueError(f"Unsupported model name for determining caption format: {model_name}")


def create_training_model(model_config_path: str) -> TrainingModelBundle:
    raw_config = load_yaml_file(model_config_path)
    model_type = raw_config.get("model_type", "viclip_ot")

    if model_type == "viclip_ot":
        raw_config.pop("model_type", None)
        config = ViCLIPOTConfig.model_validate(raw_config)
        model = ViCLIPOT(config=config)
        return TrainingModelBundle(
            model=model,
            config=config,
            tokenizer=model.text_encoder.tokenizer,
            max_length=config.max_length,
            caption_format=_get_viclip_caption_format(config.text_config.model_name),
            image_mean=C.IMAGENET_DEFAULT_MEAN,
            image_std=C.IMAGENET_DEFAULT_STD,
            required_image_size=None,
            backbone_prefixes=("text_encoder.encoder.", "image_encoder.trunk."),
            adapter_prefixes=(
                "text_encoder.dense.",
                "text_encoder.fc.",
                "image_encoder.head.",
                "logit_scale",
                "logit_bias",
            ),
            supports_tower_locking=True,
            save_trainable_state_only=False,
            test_only_uses_eval_criterion=False,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")
