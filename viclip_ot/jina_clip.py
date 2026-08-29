# pyright: reportPrivateImportUsage=false

import importlib.util
import math
from typing import Annotated, Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel, AutoTokenizer

from viclip_ot.utils.logger import logger


class JinaCLIPLoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: Annotated[int, Field(gt=0)] = 16
    alpha: Annotated[float, Field(gt=0)] = 32.0
    dropout: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0
    gradient_checkpointing: bool = True
    image_micro_batch_size: Annotated[int, Field(gt=0)] | None = None


class JinaCLIPV2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: Literal["jina_clip_v2"]
    model_name: str = "jinaai/jina-clip-v2"
    revision: str | None = None
    code_revision: str | None = None
    max_length: Annotated[int, Field(gt=0)] = 128
    image_size: Annotated[int, Field(gt=0)] = 512
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    initial_temperature: Annotated[float, Field(gt=0)] | None = None
    logit_bias: float | None = -9.0
    use_text_flash_attention: bool = True
    use_vision_xformers: bool = True
    lora: JinaCLIPLoRAConfig = JinaCLIPLoRAConfig()


class LoRAWeightParametrization(nn.Module):
    def __init__(
        self,
        *,
        fan_in: int,
        fan_out: int,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, fan_in, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(fan_out, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = dropout

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRAWeightParametrization":
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in=fan_in,
            fan_out=fan_out,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=layer.weight.device,
        )

    def forward(self, base_weight: Tensor) -> Tensor:
        lora_A = Fun.dropout(self.lora_A, p=self.dropout, training=self.training)
        update = (self.lora_B @ lora_A) * self.scaling
        return base_weight + update.to(dtype=base_weight.dtype)


def _resize_jina_vision_tower(jina_model: Any, target_image_size: int) -> None:
    vision_model = jina_model.vision_model
    vision_config = jina_model.config.vision_config
    patch_size = int(vision_config.patch_size)
    source_image_size = int(vision_config.image_size)
    if target_image_size == source_image_size:
        return
    if target_image_size % patch_size != 0:
        raise ValueError(
            f"Resized Jina image size {target_image_size} must be divisible by "
            f"patch size {patch_size}."
        )

    source_grid_size = source_image_size // patch_size
    target_grid_size = target_image_size // patch_size
    position_embeddings: Tensor | None = vision_model.pos_embed
    if position_embeddings is not None:
        expected_source_tokens = source_grid_size**2 + 1
        if position_embeddings.shape[1] != expected_source_tokens:
            raise RuntimeError(
                f"Expected {expected_source_tokens} Jina vision position tokens for "
                f"{source_image_size}x{source_image_size}, found {position_embeddings.shape[1]}."
            )
        class_position_embedding = position_embeddings[:, :1]
        patch_position_embeddings = position_embeddings[:, 1:].reshape(
            1,
            source_grid_size,
            source_grid_size,
            position_embeddings.shape[-1],
        )
        patch_position_embeddings = patch_position_embeddings.permute(0, 3, 1, 2)
        resized_patch_position_embeddings = Fun.interpolate(
            patch_position_embeddings.float(),
            size=(target_grid_size, target_grid_size),
            mode="bicubic",
            align_corners=False,
        )
        resized_patch_position_embeddings = resized_patch_position_embeddings.permute(
            0, 2, 3, 1
        ).reshape(1, target_grid_size**2, position_embeddings.shape[-1])
        resized_position_embeddings = torch.cat(
            (
                class_position_embedding.float(),
                resized_patch_position_embeddings,
            ),
            dim=1,
        ).to(device=position_embeddings.device, dtype=position_embeddings.dtype)
        vision_model.pos_embed = nn.Parameter(
            resized_position_embeddings,
            requires_grad=position_embeddings.requires_grad,
        )

    if vision_model.rel_pos_bias is not None:
        raise NotImplementedError("Resizing Jina shared relative position bias is not supported.")
    for block in vision_model.blocks:
        if block.attn.relative_position_bias_table is not None:
            raise NotImplementedError(
                "Resizing Jina per-block relative position bias is not supported."
            )

    if vision_model.rope is not None:
        source_rope = vision_model.rope
        source_frequency_cosines: Tensor = source_rope.freqs_cos
        num_heads = int(vision_config.width) // int(vision_config.head_width)
        rotary_dimension = int(vision_config.width) // num_heads // 2
        resized_rope = type(source_rope)(
            dim=rotary_dimension,
            pt_seq_len=int(vision_config.pt_hw_seq_len),
            ft_seq_len=target_grid_size,
            patch_dropout=float(vision_config.patch_dropout),
        ).to(
            device=source_frequency_cosines.device,
            dtype=source_frequency_cosines.dtype,
        )
        if resized_rope.freqs_cos.shape[0] != target_grid_size**2:
            raise RuntimeError(
                f"Expected {target_grid_size**2} resized Jina RoPE positions, "
                f"found {resized_rope.freqs_cos.shape[0]}."
            )
        vision_model.rope = resized_rope
        for block in vision_model.blocks:
            block.attn.rope = resized_rope

    target_grid_shape = (target_grid_size, target_grid_size)
    vision_model.image_size = target_image_size
    vision_model.patch_embed.img_size = (target_image_size, target_image_size)
    vision_model.patch_embed.patch_shape = target_grid_shape
    vision_model.patch_embed.num_patches = target_grid_size**2
    vision_config.image_size = target_image_size

    logger.info(
        f"Resized Jina EVA vision tower from {source_image_size}x{source_image_size} "
        f"({source_grid_size}x{source_grid_size} patches) to "
        f"{target_image_size}x{target_image_size} "
        f"({target_grid_size}x{target_grid_size} patches)."
    )


class JinaCLIPV2LoRA(nn.Module):
    def __init__(self, config: JinaCLIPV2Config) -> None:
        super().__init__()
        self.config = config

        if not torch.cuda.is_available():
            raise RuntimeError("Jina CLIP v2 LoRA training requires a CUDA GPU.")
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The Jina config requests bfloat16, but the selected CUDA GPU does not support it."
            )
        if config.use_text_flash_attention and importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "Jina CLIP v2 LoRA training requires flash-attn. Install the project optional "
                "dependency with `UV_TORCH_BACKEND=cu128 uv sync --group flash-attn`."
            )
        if config.use_vision_xformers and importlib.util.find_spec("xformers") is None:
            raise RuntimeError(
                "Jina CLIP v2 LoRA training requires xFormers. "
                "Install the project optional dependency with "
                "`UV_TORCH_BACKEND=cu128 uv sync --group jina-training`."
            )

        model_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

        remote_config = AutoConfig.from_pretrained(
            config.model_name,
            revision=config.revision,
            code_revision=config.code_revision,
            trust_remote_code=True,
        )
        remote_config.use_text_flash_attn = config.use_text_flash_attention
        remote_config.use_vision_xformers = config.use_vision_xformers

        self.jina_model: Any = AutoModel.from_pretrained(
            config.model_name,
            config=remote_config,
            revision=config.revision,
            code_revision=config.code_revision,
            trust_remote_code=True,
            dtype=model_dtype,
        )
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.revision,
            code_revision=config.code_revision,
            trust_remote_code=True,
        )

        _resize_jina_vision_tower(self.jina_model, config.image_size)

        for parameter in self.jina_model.parameters():
            parameter.requires_grad = False

        self._register_lora_adapters()
        # Keep the scalar in float32. BF16 spacing near logit_scale=4 is 0.03125,
        # much larger than the optimizer's per-step updates at a 1e-4 learning rate.
        initial_logit_scale = (
            self.jina_model.logit_scale.detach().float()
            if config.initial_temperature is None
            else torch.tensor(
                math.log(1 / config.initial_temperature),
                device=self.jina_model.logit_scale.device,
                dtype=torch.float32,
            )
        )
        self.jina_model.logit_scale = nn.Parameter(initial_logit_scale)

        self.logit_bias = None
        if config.logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.tensor(config.logit_bias, dtype=torch.float32))

        self._log_parameter_counts()
        if config.lora.image_micro_batch_size is not None:
            logger.info(
                f"Jina vision forward microbatch size: {config.lora.image_micro_batch_size}."
            )

    @property
    def logit_scale(self) -> nn.Parameter:
        return self.jina_model.logit_scale

    def _add_lora(self, layer: nn.Linear) -> None:
        parametrize.register_parametrization(
            layer,
            "weight",
            LoRAWeightParametrization.from_linear(
                layer,
                rank=self.config.lora.rank,
                alpha=self.config.lora.alpha,
                dropout=self.config.lora.dropout,
            ),
        )

    def _register_lora_adapters(self) -> None:
        text_targets: list[str] = []
        for module_name, module in self.jina_model.text_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if module_name.endswith(("mixer.Wqkv", "mixer.out_proj")):
                self._add_lora(module)
                text_targets.append(module_name)

        vision_targets: list[str] = []
        for module_name, module in self.jina_model.vision_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if module_name.endswith(("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj")):
                self._add_lora(module)
                vision_targets.append(module_name)

        expected_text_targets = 48
        expected_vision_targets = 96
        if len(text_targets) != expected_text_targets:
            raise RuntimeError(
                f"Expected {expected_text_targets} Jina text LoRA targets, "
                f"but found {len(text_targets)}. The remote model structure may have changed."
            )
        if len(vision_targets) != expected_vision_targets:
            raise RuntimeError(
                f"Expected {expected_vision_targets} Jina vision LoRA targets, "
                f"but found {len(vision_targets)}. The remote model structure may have changed."
            )

        logger.info(
            f"Registered LoRA on {len(text_targets)} text attention projections and "
            f"{len(vision_targets)} vision attention projections."
        )

    def _log_parameter_counts(self) -> None:
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        text_lora_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("jina_model.text_model")
        )
        vision_lora_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("jina_model.vision_model")
        )
        logger.info(
            f"Jina CLIP v2 parameters: total={total_parameters:,}, "
            f"trainable={trainable_parameters:,}, text_lora={text_lora_parameters:,}, "
            f"vision_lora={vision_lora_parameters:,}."
        )

    def _get_image_features(self, pixel_values: Tensor) -> Tensor:
        return cast(Tensor, self.jina_model.get_image_features(pixel_values=pixel_values))

    def _get_text_features(self, input_ids: Tensor) -> Tensor:
        return cast(Tensor, self.jina_model.get_text_features(input_ids=input_ids))

    def encode_image(self, images: Tensor, normalize: bool = False) -> Tensor:
        feature_chunks = []
        split_size_or_sections = self.config.lora.image_micro_batch_size
        if split_size_or_sections is None:
            split_size_or_sections = images.shape[0]
        for image_chunk in images.split(self.config.lora.image_micro_batch_size):
            if (
                self.config.lora.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                chunk_features = cast(
                    Tensor,
                    checkpoint(self._get_image_features, image_chunk, use_reentrant=False),
                )
            else:
                chunk_features = self._get_image_features(image_chunk)
            feature_chunks.append(chunk_features)

        features = torch.cat(feature_chunks, dim=0)
        if normalize:
            features = Fun.normalize(features, p=2, dim=-1)
        return features

    def encode_text(self, text_inputs: Any, normalize: bool = False) -> Tensor:
        input_ids: Tensor = text_inputs["input_ids"]
        if self.config.lora.gradient_checkpointing and self.training and torch.is_grad_enabled():
            features = cast(
                Tensor,
                checkpoint(self._get_text_features, input_ids, use_reentrant=False),
            )
        else:
            features = self._get_text_features(input_ids)
        if normalize:
            features = Fun.normalize(features, p=2, dim=-1)
        return features

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        vision_config = self.jina_model.config.vision_config
        return {
            "format_version": 1,
            "model_type": self.config.model_type,
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "code_revision": self.config.code_revision,
            "image_size": self.config.image_size,
            "patch_size": int(vision_config.patch_size),
            "position_interpolation": "bicubic_align_corners_false_v1",
            "lora_rank": self.config.lora.rank,
            "lora_alpha": self.config.lora.alpha,
            "lora_dropout": self.config.lora.dropout,
            "text_target_suffixes": ["mixer.Wqkv", "mixer.out_proj"],
            "vision_target_suffixes": [
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "attn.proj",
            ],
        }

    def validate_checkpoint_metadata(self, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            if self.config.image_size == 512:
                logger.warning(
                    "Loading a legacy Jina adapter checkpoint without model metadata. "
                    "Assuming the native 512x512 vision configuration."
                )
                return
            raise RuntimeError(
                "The Jina adapter checkpoint has no model metadata, so it cannot be safely "
                f"loaded into the resized {self.config.image_size}x{self.config.image_size} "
                "vision tower. Use a checkpoint saved by the configurable-resolution code."
            )

        expected_metadata = self.get_checkpoint_metadata()
        if metadata != expected_metadata:
            mismatches = {
                key: {"expected": expected_metadata.get(key), "provided": metadata.get(key)}
                for key in sorted(set(expected_metadata) | set(metadata))
                if expected_metadata.get(key) != metadata.get(key)
            }
            raise RuntimeError(
                "Jina adapter checkpoint metadata does not match the current model config: "
                f"{mismatches}."
            )

    def get_checkpoint_state_dict(self) -> dict[str, Tensor]:
        trainable_parameter_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value
            for name, value in self.state_dict().items()
            if name in trainable_parameter_names
        }

    def load_checkpoint_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        expected_names = set(self.get_checkpoint_state_dict())
        provided_names = set(state_dict)
        if provided_names != expected_names:
            missing_names = sorted(expected_names - provided_names)
            unexpected_names = sorted(provided_names - expected_names)
            raise RuntimeError(
                "Jina adapter checkpoint keys do not match the current configuration. "
                f"Missing: {missing_names}; unexpected: {unexpected_names}."
            )
        self.load_state_dict(state_dict, strict=False)

    def forward(self, images: Tensor, text_inputs: Any) -> dict[str, Tensor]:
        output = {
            "image_features": self.encode_image(images, normalize=True),
            "text_features": self.encode_text(text_inputs, normalize=True),
            "logit_scale": self.logit_scale.exp(),
        }
        if self.logit_bias is not None:
            output["logit_bias"] = self.logit_bias
        return output
