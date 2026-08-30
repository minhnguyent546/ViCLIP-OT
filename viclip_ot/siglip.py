from typing import Annotated, Any, Literal, cast

import torch.nn as nn
import torch.nn.functional as Fun
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from transformers import AutoTokenizer, SiglipModel

from viclip_ot.utils.logger import logger


class SigLIPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: Literal["siglip"]
    model_name: Literal["google/siglip-base-patch16-256-multilingual"] = (
        "google/siglip-base-patch16-256-multilingual"
    )
    revision: str = "8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6"
    max_length: Literal[64] = 64
    image_size: Literal[224, 256] = 256
    gradient_checkpointing: bool = True
    attention_implementation: Literal["eager", "sdpa"] = "sdpa"
    logit_scale_min: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.01
    logit_scale_max: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 200.0

    @model_validator(mode="after")
    def validate_logit_scale_bounds(self) -> "SigLIPConfig":
        if self.logit_scale_min >= self.logit_scale_max:
            raise ValueError("logit_scale_min must be less than logit_scale_max")
        return self


class SigLIP(nn.Module):
    def __init__(self, config: SigLIPConfig) -> None:
        super().__init__()
        self.config = config
        self.siglip_model = SiglipModel.from_pretrained(
            config.model_name,
            revision=config.revision,
            attn_implementation=config.attention_implementation,
        )
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.revision,
        )

        native_image_size = int(self.siglip_model.config.vision_config.image_size)
        patch_size = int(self.siglip_model.config.vision_config.patch_size)
        if config.image_size % patch_size != 0:
            raise ValueError(
                f"SigLIP image size {config.image_size} must be divisible by patch size "
                f"{patch_size}."
            )
        self.interpolate_pos_encoding = config.image_size != native_image_size
        if self.interpolate_pos_encoding:
            logger.info(
                f"SigLIP will interpolate vision position embeddings from "
                f"{native_image_size // patch_size}x{native_image_size // patch_size} to "
                f"{config.image_size // patch_size}x{config.image_size // patch_size}."
            )

        if config.gradient_checkpointing:
            self.siglip_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        logger.info(
            f"SigLIP parameters: total={total_parameters:,}, trainable={trainable_parameters:,}."
        )

    @property
    def logit_scale(self) -> nn.Parameter:
        return self.siglip_model.logit_scale

    @property
    def logit_bias(self) -> nn.Parameter:
        return self.siglip_model.logit_bias

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        vision_config = self.siglip_model.config.vision_config
        return {
            "format_version": 1,
            "model_type": self.config.model_type,
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "image_size": self.config.image_size,
            "native_image_size": int(vision_config.image_size),
            "patch_size": int(vision_config.patch_size),
            "position_interpolation": (
                "bicubic_align_corners_false_dynamic_v1"
                if self.interpolate_pos_encoding
                else "none"
            ),
            "max_length": self.config.max_length,
            "text_padding": "max_length",
        }

    def validate_checkpoint_metadata(self, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            if self.config.image_size == int(self.siglip_model.config.vision_config.image_size):
                logger.warning(
                    "Loading a legacy SigLIP checkpoint without model metadata. "
                    "Assuming the native image resolution."
                )
                return
            raise RuntimeError(
                "The SigLIP checkpoint has no model metadata, so it cannot be safely loaded "
                f"under the interpolated {self.config.image_size}x{self.config.image_size} "
                "resolution protocol."
            )

        expected_metadata = self.get_checkpoint_metadata()
        if metadata != expected_metadata:
            mismatches = {
                key: {"expected": expected_metadata.get(key), "provided": metadata.get(key)}
                for key in sorted(set(expected_metadata) | set(metadata))
                if expected_metadata.get(key) != metadata.get(key)
            }
            raise RuntimeError(
                "SigLIP checkpoint metadata does not match the current model config: "
                f"{mismatches}."
            )

    def encode_image(self, images: Tensor, normalize: bool = False) -> Tensor:
        features = cast(
            Tensor,
            self.siglip_model.get_image_features(
                pixel_values=images,  # pyright: ignore[reportArgumentType]
                interpolate_pos_encoding=self.interpolate_pos_encoding,
            ),
        )
        if normalize:
            features = Fun.normalize(features, p=2, dim=-1)
        return features

    def encode_text(self, text_inputs: Any, normalize: bool = False) -> Tensor:
        features = cast(Tensor, self.siglip_model.get_text_features(**text_inputs))
        if normalize:
            features = Fun.normalize(features, p=2, dim=-1)
        return features

    def forward(self, images: Tensor, text_inputs: Any) -> dict[str, Tensor]:
        return {
            "image_features": self.encode_image(images, normalize=True),
            "text_features": self.encode_text(text_inputs, normalize=True),
            "logit_scale": self.logit_scale.exp(),
            "logit_bias": self.logit_bias,
        }
