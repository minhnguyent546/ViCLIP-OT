from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Literal

import torch
from transformers import PretrainedConfig, logging

logger = logging.get_logger(__name__)


class ViCLIPOTTextConfig(PretrainedConfig):
    model_type = "viclip_ot_text"

    _SUPPORTED_MODELS = [
        "google/embeddinggemma-300m",
        "baai/bge-m3",
        "qwen/qwen3-embedding-0.6b",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
        "keepitreal/vietnamese-sbert",
    ]

    def __init__(
        self,
        model_name: str = "keepitreal/vietnamese-sbert",
        proj: Literal["linear", "none"] = "none",
        proj_bias: bool = False,
        **kwargs,
    ) -> None:
        # Legacy training configs may include this key; it is not part of HF config schema.
        kwargs.pop("pretrained", None)
        super().__init__(**kwargs)

        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. Call `ViCLIPOTTextConfig.list_models()` to see supported models."
            )

        self.model_name = model_name
        self.proj = proj
        self.proj_bias = proj_bias

    @classmethod
    def list_models(cls) -> None:
        """Lists the supported text models."""
        logger.info("Supported text models:")
        for model in cls._SUPPORTED_MODELS:
            logger.info(f"- {model}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> ViCLIPOTTextConfig:
        cls._set_token_in_kwargs(kwargs=kwargs, token=token)

        configdict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            revision=revision,
            **kwargs,
        )

        # get the vision config dict if we are loading from ViCLIPOTConfig
        if configdict.get("model_type") == "viclip_ot":
            configdict = configdict["text_config"]
        if (
            "model_type" in configdict
            and hasattr(cls, "model_type")
            and configdict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {configdict['model_type']} to "
                f"instantiate a model of type {cls.model_type}. This is not supported "
                "for all configurations of models and can yield errors."
            )

        return cls.from_dict(configdict, **kwargs)


class ViCLIPOTVisionConfig(PretrainedConfig):
    model_type = "viclip_ot_vision"

    _SUPPORTED_MODELS = [
        "timm/convnext_base.dinov3_lvd1689m",
        "timm/convnext_small.dinov3_lvd1689m",
        "timm/convnext_base.fb_in22k_ft_in1k",
        "timm/convnextv2_base.fcmae_ft_in22k_in1k",
        "timm/vit_base_patch16_dinov3.lvd1689m",
        "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "timm/vit_large_patch16_dinov3.lvd1689m",
    ]

    def __init__(
        self,
        model_name: str = "timm/vit_base_patch16_dinov3.lvd1689m",
        pool: Literal["avg", "max", "abs_attn", "rot_attn", ""] = "avg",
        proj: Literal["linear", "mlp"] = "mlp",
        proj_bias: bool = False,
        proj_dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        # Legacy training configs may include this key; it is not part of HF config schema.
        kwargs.pop("pretrained", None)
        super().__init__(**kwargs)

        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. Call `ViCLIPOTVisionConfig.list_models()` to see supported models."
            )

        self.model_name = model_name
        self.pool = pool
        self.proj = proj
        self.proj_bias = proj_bias
        self.proj_dropout_rate = proj_dropout_rate

    @classmethod
    def list_models(cls) -> None:
        """Lists the supported vision models."""
        logger.info("Supported vision models:")
        for model in cls._SUPPORTED_MODELS:
            logger.info(f"- {model}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> ViCLIPOTVisionConfig:
        cls._set_token_in_kwargs(kwargs=kwargs, token=token)

        configdict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            revision=revision,
            **kwargs,
        )

        # get the vision config dict if we are loading from ViCLIPOTConfig
        if configdict.get("model_type") == "viclip_ot":
            configdict = configdict["vision_config"]
        if (
            "model_type" in configdict
            and hasattr(cls, "model_type")
            and configdict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {configdict['model_type']} to "
                f"instantiate a model of type {cls.model_type}. This is not supported "
                "for all configurations of models and can yield errors."
            )

        return cls.from_dict(configdict, **kwargs)


class ViCLIPOTConfig(PretrainedConfig):
    model_type = "viclip_ot"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        embed_dim: int = 768,
        initial_temperature: float = 0.07,
        logit_bias: float | None = None,
        dtype: str | torch.dtype | None = None,
        **kwargs,
    ) -> None:
        text_config_dict: dict[str, Any] | None = kwargs.pop("text_config_dict", None)
        vision_config_dict: dict[str, Any] | None = kwargs.pop("vision_config_dict", None)

        self.embed_dim = embed_dim
        self.initial_temperature = initial_temperature
        self.logit_bias = logit_bias

        super().__init__(**kwargs)

        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            _text_config_dict = ViCLIPOTTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and
            # `text_config` but being different.
            for key, value in _text_config_dict.items():
                if (
                    key in text_config
                    and value != text_config[key]
                    and key not in ["transformers_version"]
                ):
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and "
                            f"`text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used '
                            f"instead."
                        )
                    # If inferred from default argument values (
                    # just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to "
                            f"initialize `ViCLIPOTTextConfig`. The "
                            f'value `text_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = ViCLIPOTVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict`
            # and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if (
                    key in vision_config
                    and value != vision_config[key]
                    and key not in ["transformers_version"]
                ):
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and "
                            f"`vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will '
                            f"be used instead."
                        )
                    # If inferred from default argument values
                    # (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to "
                            f"initialize `ViCLIPOTVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in
            # `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = {}
            logger.info(
                "`text_config` is `None`. Initializing the `ViCLIPOTTextConfig` with "
                "default values."
            )

        if vision_config is None:
            vision_config = {}
            logger.info(
                "`vision_config` is `None`. initializing the `ViCLIPOTVisionConfig` "
                "with default values."
            )

        self.text_config = ViCLIPOTTextConfig(**text_config)
        self.vision_config = ViCLIPOTVisionConfig(**vision_config)

        if (
            isinstance(dtype, str)
            and hasattr(torch, dtype)
            and type(getattr(torch, dtype)) is torch.dtype
        ):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: ViCLIPOTTextConfig,
        vision_config: ViCLIPOTVisionConfig,
        **kwargs,
    ) -> ViCLIPOTConfig:
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        output = deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
