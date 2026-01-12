# pyright: reportAssignmentType=false
import json
from typing import Any, Literal, OrderedDict

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file as load_safetensors
from timm.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
from timm.layers.attention_pool2d import RotAttentionPool2d
from timm.layers.mlp import Mlp
from timm.models.helpers import group_modules, group_parameters
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from viclip_ot.utils.logger import logger
from viclip_ot.utils.training import freeze_batch_norm_2d


class ViCLIPOTImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    pretrained: bool = True
    pool: Literal["avg", "max", "abs_attn", "rot_attn", ""] = "avg"
    proj: Literal["linear", "mlp"] = "mlp"
    proj_bias: bool = False
    proj_dropout_rate: float = 0.0


class ViCLIPOTTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str


class ViCLIPOTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_config: ViCLIPOTImageConfig
    text_config: ViCLIPOTTextConfig
    embed_dim: int
    logit_scale: float = np.log(1 / 0.07)
    logit_bias: float | None = None


class ImageEncoder(nn.Module):
    _SUPPORTED_MODELS = ["timm/convnext_base.dinov3_lvd1689m"]

    def __init__(
        self,
        config: ViCLIPOTImageConfig,
        *,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        if self.config.model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {self.config.model_name}. Call `list_models()` to see supported models."
            )

        is_custom_pool = self.config.pool in ("abs_attn", "rot_attn")
        self.trunk = timm.create_model(
            model_name=self.config.model_name,
            pretrained=self.config.pretrained,
        )
        trunk_default_config: dict[str, Any] = self.trunk.default_cfg
        assert "pool_size" in trunk_default_config
        feat_size = trunk_default_config["pool_size"]
        if is_custom_pool:
            # if attn pooling used, remove both classifier and default pool
            self.trunk.reset_classifier(num_classes=0, global_pool="")  # pyright: ignore[reportCallIssue]
        else:
            # reset global pool if pool config set, otherwise leave as network default
            reset_kwargs = {"global_pool": self.config.pool} if self.config.pool else {}
            self.trunk.reset_classifier(0, **reset_kwargs)  # pyright: ignore[reportCallIssue]

        prev_chs: int = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if self.config.pool == "abs_attn":
            head_layers["pool"] = AbsAttentionPool2d(
                in_features=prev_chs, feat_size=feat_size, out_features=embed_dim
            )
            prev_chs = embed_dim
        elif self.config.pool == "rot_attn":
            head_layers["pool"] = RotAttentionPool2d(in_features=prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if self.config.proj == "linear":
            head_layers["drop"] = nn.Dropout(self.config.proj_dropout_rate)
            proj_layer = nn.Linear(
                in_features=prev_chs, out_features=embed_dim, bias=self.config.proj_bias
            )
            # Initialize with Xavier/Glorot for better gradient flow in contrastive learning
            nn.init.xavier_uniform_(proj_layer.weight)
            if proj_layer.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(proj_layer.bias)
            head_layers["proj"] = proj_layer
        elif self.config.proj == "mlp":
            head_layers["mlp"] = Mlp(
                in_features=prev_chs,
                hidden_features=2 * embed_dim,
                out_features=embed_dim,
                drop=(self.config.proj_dropout_rate, 0),
                bias=(True, self.config.proj_bias),
            )
        else:
            raise ValueError(f"Unsupported proj type: {self.config.proj}")

        self.head = nn.Sequential(head_layers)

    def freeze(self, last_unfreeze_groups: int = 0, freeze_bn_stats: bool = False):
        """Freeze trunk, leave the last `last_unfreeze_groups` unfreeze.

        Adapted from: https://github.com/mlfoundations/open_clip/blob/d3cdb734a2710feeb4c6307df037afa5f786a3e1/src/open_clip/timm_model.py#L105.
        """
        if not last_unfreeze_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            matcher = self.trunk.group_matcher()  # pyright: ignore[reportCallIssue]
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - last_unfreeze_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @classmethod
    def list_models(cls) -> list[str]:
        return cls._SUPPORTED_MODELS

    def forward(self, x: Tensor) -> Tensor:
        y = self.trunk(x)
        y = self.head(y)

        return y


class TextEncoder(nn.Module):
    _SUPPORTED_MODELS = ["google/embeddinggemma-300m"]

    def __init__(
        self,
        config: ViCLIPOTTextConfig,
        *,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        if self.config.model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {self.config.model_name}. Call `list_models()` to see supported models."
            )

        if self.config.model_name == "google/embeddinggemma-300m":
            self._prepare_embeddinggemma_300m()
        else:
            raise NotImplementedError(
                f"TextEncoder for model {self.config.model_name} is not implemented yet."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

        # TODO: works for Gemma3, make more general
        intern_embed_dim = self.encoder.config.hidden_size
        assert intern_embed_dim is not None, "Failed to get sentence embedding dimension."
        self.fc = nn.Linear(intern_embed_dim, embed_dim)

        # Initialize the projection layer with Xavier/Glorot initialization
        # to help with gradient flow in contrastive learning
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            nn.init.zeros_(self.fc.bias)

    @classmethod
    def list_models(cls) -> list[str]:
        return cls._SUPPORTED_MODELS

    def _prepare_embeddinggemma_300m(self) -> None:
        """Download pre-trained weights and sentence-transformers stuff
        to be able to use the model with Hugging Face."""
        if not self.config.model_name == "google/embeddinggemma-300m":
            raise ValueError(
                "_prepare_embeddinggemma_300m is only applicable for 'google/embeddinggemma-300m' model."
            )

        # discover module paths
        modules_path = hf_hub_download(self.config.model_name, filename="modules.json")
        with open(modules_path, "r", encoding="utf-8") as f:
            modules = json.load(f)

        xf_sub = next(m["path"] for m in modules if "Transformer" in m["type"])
        pool_sub = next(m["path"] for m in modules if "Pooling" in m["type"])
        dense_subs = [m["path"] for m in modules if "Dense" in m["type"]]
        norm_exists = any("Normalize" in m["type"] for m in modules)

        logger.info(f"[TextEncoder - embeddinggemma-300m] Transformer subfolder: {xf_sub}")
        logger.info(f"[TextEncoder - embeddinggemma-300m] Pooling subfolder: {pool_sub}")
        logger.info(f"[TextEncoder - embeddinggemma-300m] Dense subfolders: {dense_subs}")
        logger.info(f"[TextEncoder - embeddinggemma-300m] Has Normalize: {norm_exists}")

        # load backbone and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)

        self.dense = nn.Sequential(*[self._load_dense(ds) for ds in sorted(dense_subs)])

    def _load_dense(self, subfolder: str) -> nn.Module:
        cfg_p = hf_hub_download(self.config.model_name, filename=f"{subfolder}/config.json")
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        lin = torch.nn.Linear(cfg["in_features"], cfg["out_features"], bias=cfg.get("bias", True))

        # load weights (safetensors preferred; fall back to bin)
        try:
            st = load_safetensors(
                hf_hub_download(self.config.model_name, filename=f"{subfolder}/model.safetensors")
            )
        except Exception:
            st = torch.load(
                hf_hub_download(self.config.model_name, filename=f"{subfolder}/pytorch_model.bin"),
                map_location="cpu",
            )

        # Normalize key names if needed
        if "linear.weight" in st:
            st["weight"] = st.pop("linear.weight")
        if "linear.bias" in st:
            st["bias"] = st.pop("linear.bias")

        # Ensure weights are compute dtype
        lin.load_state_dict(st, strict=True)

        act = cfg.get("activation_function", None)
        if "Tanh" in act:
            activation_fun = nn.Tanh()
        elif "ReLU" in act:
            activation_fun = nn.ReLU()
        elif "Identity" in act:
            activation_fun = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        return nn.Sequential(lin, activation_fun)

    def get_embeddings(self, inputs: dict[str, Tensor], *, normalize: bool = False) -> Tensor:
        # forward Pass
        outputs = self.encoder(**inputs)

        # get Last Hidden State (batch_size, seq_len, hidden_dim)
        last_hidden_state = outputs.last_hidden_state

        # We must mask out padding tokens so they don't affect the average
        attention_mask = inputs["attention_mask"]

        # Expand mask to match hidden state dimensions: (batch, seq_len) -> (batch, seq_len, hidden_dim)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .to(last_hidden_state.dtype)
        )

        # Sum embeddings ignoring padding
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask (clamp to avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1).float(), min=1e-9)

        # Calculate mean in float32, then cast back to original dtype
        embeddings = (sum_embeddings.float() / sum_mask).to(last_hidden_state.dtype)

        # pass through dense layers
        embeddings = self.dense(embeddings)

        # normalize (L2 Norm)
        if normalize:
            embeddings = Fun.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(self, inputs, normalize: bool = False) -> Tensor:
        y = self.get_embeddings(inputs, normalize=normalize)
        y = self.fc(y)

        return y


# TODO: add logit_scale
class ViCLIPOT(nn.Module):
    """ViCLIP-OT model."""

    def __init__(self, config: ViCLIPOTConfig) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(
            config=config.image_config,
            embed_dim=config.embed_dim,
        )
        self.text_encoder = TextEncoder(config=config.text_config, embed_dim=config.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale, dtype=torch.float32))
        self.logit_bias = None
        if config.logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.tensor(config.logit_bias, dtype=torch.float32))

    def lock_image_tower(self, last_unfreeze_groups: int = 0, freeze_bn_stats: bool = False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.image_encoder.freeze(
            last_unfreeze_groups=last_unfreeze_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        assert freeze_layer_norm, (
            "Unfreezing LayerNorm is not supported. LayerNorm treated like other weights."
        )
        raise NotImplementedError("TODO: Locking text tower is not implemented yet.")

    def encode_image(self, image, normalize: bool = False):
        features = self.image_encoder(image)
        if normalize:
            features = Fun.normalize(features, p=2, dim=-1)

        return features

    def encode_text(self, inputs, normalize: bool = False):
        features = self.text_encoder(inputs=inputs, normalize=normalize)

        return features

    def forward(self, images: Tensor, text_inputs):
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(text_inputs, normalize=True)

        output_dict = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
        }
        if self.logit_bias is not None:
            output_dict["logit_bias"] = self.logit_bias

        return output_dict
