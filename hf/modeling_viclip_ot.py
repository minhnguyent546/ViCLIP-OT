import base64
import json
from dataclasses import dataclass
from io import BytesIO
from logging import DEBUG, INFO
from typing import Any, OrderedDict

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
from timm.layers.attention_pool2d import RotAttentionPool2d
from timm.layers.mlp import Mlp
from timm.models import group_modules, group_parameters
from torchvision.ops.misc import FrozenBatchNorm2d
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
    PreTrainedModel,
    logging,
)
from transformers.utils.generic import ModelOutput

try:
    import timm
except ImportError as err:
    raise ImportError(
        "timm library is required for ViCLIPOTVisionModel. Please install it with `pip install timm`."
    ) from err

try:
    from tqdm.autonotebook import trange

    has_tqdm = True
except ImportError:
    trange = None
    has_tqdm = False

from .configuration_viclip_ot import ViCLIPOTConfig, ViCLIPOTTextConfig, ViCLIPOTVisionConfig
from .processing_viclip_ot import (
    ViCLIPOTProcessor,
)

logger = logging.get_logger(__name__)


@dataclass
class ViCLIPOTTextModelOutput(ModelOutput):
    text_features: torch.Tensor | None = None


@dataclass
class ViCLIPOTVisionModelOutput(ModelOutput):
    image_features: torch.Tensor | None = None


@dataclass
class ViCLIPOTOutput(ModelOutput):
    text_features: torch.Tensor | None = None
    image_features: torch.Tensor | None = None
    logits_per_text: torch.Tensor | None = None
    logits_per_image: torch.Tensor | None = None
    logit_scale: torch.Tensor | None = None
    logit_bias: torch.Tensor | None = None
    loss: torch.Tensor | None = None


def freeze_batch_norm_2d(module, module_match=None, name=""):
    """Taken from: https://github.com/mlfoundations/open_clip/blob/d3cdb734a2710feeb4c6307df037afa5f786a3e1/src/open_clip/utils.py

    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    if module_match is None:
        module_match = {}

    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(
        module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features  # pyright: ignore
        res.affine = module.affine  # pyright: ignore
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data  # pyright: ignore
        res.running_var.data = module.running_var.data  # pyright: ignore
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


class ViCLIPOTPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for
    downloading and loading pretrained models.
    """

    config_class = ViCLIPOTConfig
    base_model_prefix = "viclip_ot"
    supports_gradient_checkpointing = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if "dtype" not in kwargs:
            kwargs["dtype"] = "auto"

        return super().from_pretrained(*args, **kwargs)


class ViCLIPOTTextModel(ViCLIPOTPretrainedModel):
    config_class = ViCLIPOTTextConfig
    base_model_prefix = "viclip_ot_text"

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
        config: ViCLIPOTTextConfig,
        *,
        embed_dim: int,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.embed_dim = embed_dim
        if self.config.model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {self.config.model_name}. Call `list_models()` to see supported models."
            )

        tokenizer_args: dict[str, Any] = {}
        if self.config.model_name == "qwen/qwen3-embedding-0.6b":
            tokenizer_args["padding_side"] = "left"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, **tokenizer_args
        )
        _model_config = AutoConfig.from_pretrained(self.config.model_name, trust_remote_code=True)
        logger.info(f"Initializing TextEncoder {self.config.model_name} architecture.")
        self.encoder = AutoModel.from_config(_model_config, trust_remote_code=True)

        # TODO: this is a messy way, infer from modules.json should be better
        if self.config.model_name == "google/embeddinggemma-300m":
            self._add_dense_for_embeddinggemma_300m()
        elif self.config.model_name == "baai/bge-m3":
            self.dense = nn.Identity()  # no extra layers needed
        elif self.config.model_name == "qwen/qwen3-embedding-0.6b":
            self.dense = nn.Identity()  # no extra layers needed
        elif self.config.model_name in (
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
        ):
            self.dense = nn.Identity()  # no extra layers needed
        elif self.config.model_name == "keepitreal/vietnamese-sbert":
            self.dense = nn.Identity()  # no extra layers needed
        else:
            raise NotImplementedError(
                f"TextEncoder for model {self.config.model_name} is not implemented yet."
            )

        # TODO: works for Gemma3, make more general
        intermediate_embed_dim = self.encoder.config.hidden_size
        assert intermediate_embed_dim is not None, "Failed to get sentence embedding dimension."

        if self.config.proj == "linear":
            self.fc = nn.Linear(intermediate_embed_dim, embed_dim, bias=self.config.proj_bias)

            # Initialize the projection layer with Xavier/Glorot initialization
            # to help with gradient flow in contrastive learning
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(self.fc.bias)
        elif self.config.proj == "none":
            if intermediate_embed_dim != embed_dim:
                raise ValueError(
                    f"intermediate_embed_dim {intermediate_embed_dim} != embed_dim {embed_dim} but `proj` is configured to `none`. Consider setting `proj` to `linear`."
                )
            self.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported proj type: {self.config.proj}")

        self.post_init()

    @classmethod
    def list_models(cls) -> list[str]:
        return cls._SUPPORTED_MODELS

    def _add_dense_for_embeddinggemma_300m(self) -> None:
        """Add dense layers for embeddinggemma-300m model."""
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

        dense_layers = [self._load_dense(ds) for ds in sorted(dense_subs)]
        if dense_layers and dense_layers[-1][-1] == nn.Identity():  # pyright: ignore[reportIndexIssue]
            dense_layers[-1][-1] = nn.GELU()  # pyright: ignore[reportIndexIssue]

        if dense_layers:
            self.dense = nn.Sequential(*dense_layers)
        else:
            self.dense = nn.Identity()

    def _load_dense(self, subfolder: str) -> nn.Module:
        cfg_p = hf_hub_download(self.config.model_name, filename=f"{subfolder}/config.json")
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        lin = torch.nn.Linear(cfg["in_features"], cfg["out_features"], bias=cfg.get("bias", True))

        nn.init.xavier_uniform_(lin.weight)
        if lin.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            nn.init.zeros_(lin.bias)

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

    def _last_token_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.config.model_name != "qwen/qwen3-embedding-0.6b":
            raise ValueError(
                "_last_token_pool is only supported for 'qwen/qwen3-embedding-0.6b' model."
            )

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
            ]

    def _average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            self.config.model_name != "intfloat/multilingual-e5-base"
            and self.config.model_name != "intfloat/multilingual-e5-large"
        ):
            raise ValueError(
                "_average_pool is only supported for 'intfloat/multilingual-e5-base' and 'intfloat/multilingual-e5-large' models."
            )

        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def freeze(self, unfreeze_dense: bool = False) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

        if not unfreeze_dense:
            if hasattr(self, "dense"):
                for param in self.dense.parameters():
                    param.requires_grad = False

    def get_embeddings(
        self, inputs: dict[str, torch.Tensor], *, normalize: bool = False
    ) -> torch.Tensor:
        assert self.config.model_name in (
            "google/embeddinggemma-300m",
            "baai/bge-m3",
            "qwen/qwen3-embedding-0.6b",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "keepitreal/vietnamese-sbert",
        ), (
            "get_embeddings currently only supports 'google/embeddinggemma-300m', 'baai/bge-m3', 'qwen/qwen3-embedding-0.6b', "
            "'intfloat/multilingual-e5-base', 'intfloat/multilingual-e5-large', and 'keepitreal/vietnamese-sbert' models."
        )

        # forward Pass
        outputs = self.encoder(**inputs)

        if self.config.model_name == "keepitreal/vietnamese-sbert":
            # https://huggingface.co/keepitreal/vietnamese-sbert
            # Mean Pooling - Take attention mask into account for correct averaging
            def __mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[
                    0
                ]  # First element of model_output contains all token embeddings
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9
                )

            sentence_embeddings = __mean_pooling(outputs, inputs["attention_mask"])

            if normalize:
                sentence_embeddings = Fun.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings

        # get Last Hidden State (batch_size, seq_len, hidden_dim)
        last_hidden_state = outputs.last_hidden_state

        if self.config.model_name == "baai/bge-m3":
            # use CLS token
            last_hidden_state = last_hidden_state[:, 0, :]

            if normalize:
                last_hidden_state = Fun.normalize(last_hidden_state, p=2, dim=1)

            return last_hidden_state

        if self.config.model_name == "qwen/qwen3-embedding-0.6b":
            # use last token pooling
            last_hidden_state = self._last_token_pool(
                last_hidden_states=last_hidden_state,
                attention_mask=inputs["attention_mask"],
            )

            if normalize:
                last_hidden_state = Fun.normalize(last_hidden_state, p=2, dim=1)

            return last_hidden_state

        if (
            self.config.model_name == "intfloat/multilingual-e5-base"
            or self.config.model_name == "intfloat/multilingual-e5-large"
        ):
            # use average pooling
            last_hidden_state = self._average_pool(
                last_hidden_states=last_hidden_state,
                attention_mask=inputs["attention_mask"],
            )

            if normalize:
                last_hidden_state = Fun.normalize(last_hidden_state, p=2, dim=1)

            return last_hidden_state

        assert self.config.model_name == "google/embeddinggemma-300m", (
            "Only 'google/embeddinggemma-300m' model reaches this point."
        )

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

    def forward(
        self,
        input_ids: torch.Tensor | BatchEncoding | BatchFeature | dict[str, torch.Tensor],
        normalize: bool = True,
        return_dict: bool | None = None,
        *_,
        **__,
    ) -> tuple[torch.Tensor | None, ...] | ViCLIPOTTextModelOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if isinstance(input_ids, torch.Tensor):
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        else:
            inputs = dict(input_ids)
            if "attention_mask" not in inputs:
                input_tensor = inputs.get("input_ids")
                if not isinstance(input_tensor, torch.Tensor):
                    raise TypeError("`input_ids` must contain a tensor at key `input_ids`.")
                inputs["attention_mask"] = torch.ones_like(input_tensor)

        text_features = self.get_embeddings(inputs=inputs, normalize=normalize)  # pyright: ignore[reportArgumentType]
        outputs = ViCLIPOTTextModelOutput(text_features=text_features)

        return outputs if return_dict else outputs.to_tuple()


class ViCLIPOTVisionModel(ViCLIPOTPretrainedModel):
    config_class = ViCLIPOTVisionConfig
    base_model_prefix = "viclip_ot_vision"
    main_input_name = "pixel_values"

    _SUPPORTED_MODELS = [
        "timm/convnext_base.dinov3_lvd1689m",
        "timm/convnext_small.dinov3_lvd1689m",
        "timm/convnextv2_base.fcmae_ft_in22k_in1k",
        "timm/vit_base_patch16_dinov3.lvd1689m",
        "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "timm/vit_large_patch16_dinov3.lvd1689m",
    ]

    def __init__(
        self,
        config: ViCLIPOTVisionConfig,
        *,
        embed_dim: int,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.embed_dim = embed_dim
        if self.config.model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {self.config.model_name}. Call `list_models()` to see supported models."
            )

        is_custom_pool = self.config.pool in ("abs_attn", "rot_attn")
        logger.info("Initializing ImageEncoder trunk architecture.")
        self.trunk = timm.create_model(
            model_name=self.config.model_name,
            pretrained=False,
        )
        trunk_default_config: dict[str, Any] = self.trunk.default_cfg  # pyright: ignore[reportAssignmentType]
        assert "pool_size" in trunk_default_config
        feat_size = trunk_default_config["pool_size"]
        if is_custom_pool:
            # if attn pooling used, remove both classifier and default pool
            self.trunk.reset_classifier(num_classes=0, global_pool="")  # pyright: ignore[reportCallIssue]

        else:
            # reset global pool if pool config set, otherwise leave as network default
            reset_kwargs = {"global_pool": self.config.pool} if self.config.pool else {}
            self.trunk.reset_classifier(0, **reset_kwargs)  # pyright: ignore[reportCallIssue]

        prev_chs: int = self.trunk.num_features  # pyright: ignore[reportAssignmentType]

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
            mlp = Mlp(
                in_features=prev_chs,
                hidden_features=2 * embed_dim,
                out_features=embed_dim,
                drop=(self.config.proj_dropout_rate, 0),
                bias=(True, self.config.proj_bias),
            )
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                        nn.init.zeros_(m.bias)
            head_layers["mlp"] = mlp
        else:
            raise ValueError(f"Unsupported proj type: {self.config.proj}")

        self.head = nn.Sequential(head_layers)

        self.post_init()

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

    def forward(
        self,
        pixel_values: torch.Tensor | BatchFeature,
        normalize: bool = True,
        return_dict: bool | None = None,
        *_,
        **__,
    ) -> tuple[torch.Tensor | None, ...] | ViCLIPOTVisionModelOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        x = pixel_values.pixel_values if isinstance(pixel_values, BatchFeature) else pixel_values

        y = self.trunk(x)
        y = self.head(y)

        if normalize:
            y = Fun.normalize(y, p=2, dim=-1)

        outputs = ViCLIPOTVisionModelOutput(image_features=y)

        return outputs if return_dict else outputs.to_tuple()


class ViCLIPOTModel(ViCLIPOTPretrainedModel):
    config_class = ViCLIPOTConfig
    base_model_prefix = "viclip_ot"

    def __init__(self, config: ViCLIPOTConfig) -> None:
        super().__init__(config)

        self.text_model = ViCLIPOTTextModel(
            self.config.text_config, embed_dim=self.config.embed_dim
        )
        self.vision_model = ViCLIPOTVisionModel(
            self.config.vision_config, embed_dim=self.config.embed_dim
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / self.config.initial_temperature), dtype=torch.float32)
        )
        self.logit_bias = None
        if config.logit_bias is not None:
            self.logit_bias = nn.Parameter(
                torch.tensor(self.config.logit_bias, dtype=torch.float32)
            )

        self._tokenizer: AutoTokenizer = self.text_model.tokenizer
        self._processor: ViCLIPOTProcessor | None = None

        self.post_init()

    @property
    def processor(self) -> ViCLIPOTProcessor:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.config.name_or_path, trust_remote_code=True
            )

        return self._processor  # pyright: ignore[reportReturnType]

    def get_text_features(
        self,
        input_ids: torch.Tensor | BatchEncoding | BatchFeature | dict[str, torch.Tensor],
        normalize: bool = False,
        return_dict: bool | None = None,
        *_,
        **__,
    ) -> torch.Tensor:
        return self.text_model(
            input_ids=input_ids, normalize=normalize, return_dict=return_dict
        ).text_features

    def get_image_features(
        self,
        pixel_values: torch.Tensor | BatchFeature,
        normalize: bool = True,
        return_dict: bool | None = None,
        *_,
        **__,
    ) -> torch.Tensor:
        pixel_values = (
            pixel_values.pixel_values if isinstance(pixel_values, BatchFeature) else pixel_values
        )
        return self.vision_model(
            pixel_values=pixel_values, normalize=normalize, return_dict=return_dict
        ).image_features

    @staticmethod
    def _decode_image_data(image_data_str: str) -> Image.Image:
        _header, data = image_data_str.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))

    @torch.inference_mode()
    def encode_image(
        self,
        images: str | list[str | Image.Image],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: torch.device | None = None,
        normalize: bool = True,
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """
        Computes image embeddings
        Args:
            images(`str` or `List[Union[str, Image.Image]]`):
                Image paths, URLs, PIL images, or data:image/ strings to be encoded
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding images. If set to None, progress bar
                is only shown when `logger.level == logging.INFO` or
                `logger.level == logging.DEBUG`
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors. Else, it is a list of
                pytorch tensors
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return. Overwrites any setting
                from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize(`bool`, *optional*, defaults to True):
                If set to true, returned vectors will have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity
                can be used

        Returns:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked
            tensor is returned. If convert_to_numpy, a numpy matrix is returned
        """

        _is_training = self.training
        self.eval()

        all_embeddings = []

        if not show_progress_bar:
            show_progress_bar = (
                logger.getEffectiveLevel() == INFO or logger.getEffectiveLevel() == DEBUG
            )
        if convert_to_tensor:
            convert_to_numpy = False

        _input_was_single_img = False
        if isinstance(images, str) or not hasattr(images, "__len__"):
            images = [images]  # pyright: ignore[reportAssignmentType]
            _input_was_single_img = True

        if device is not None:
            self.to(device)  # pyright: ignore[reportArgumentType]

        _permutation = np.argsort([-len(str(i)) for i in images])
        _inverse_permutation = np.argsort(_permutation)
        images = [images[idx] for idx in _permutation]

        if has_tqdm:
            assert trange is not None
            range_iter = trange(
                0,
                len(images),
                batch_size,
                desc="Encoding images",
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(images), batch_size)

        for i in range_iter:
            _pil_images = []
            for img in images[i : i + batch_size]:
                if isinstance(img, str):
                    if img.startswith("http"):
                        response = requests.get(img)
                        image = Image.open(BytesIO(response.content))
                    elif img.startswith("data:image/"):
                        image = self._decode_image_data(img)
                    else:
                        image = Image.open(img)
                elif isinstance(img, Image.Image):
                    image = img
                else:
                    raise ValueError("Unsupported image format")

                # handle palette images with transparency
                if image.mode == "P" and "transparency" in image.info:
                    image = image.convert("RGBA")

                image = image.convert("RGB")

                _pil_images.append(image)

            inputs = self.processor(images=_pil_images, return_tensors="pt").to(self.device)
            embeddings = self.get_image_features(pixel_values=inputs, normalize=normalize)

            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.to(torch.float32).numpy() for emb in all_embeddings])

        if _input_was_single_img:
            all_embeddings = all_embeddings[0]

        self.train(_is_training)

        return all_embeddings

    @torch.inference_mode()
    def encode_text(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: torch.device | None = None,
        normalize: bool = True,
        **tokenizer_kwargs,
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """
        Computes text embeddings
        Args:
            sentences(`str` or `List[str]`):
                Sentence or sentences to be encoded
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding sentences. If set to None, progress
                bar is only shown when `logger.level == logging.INFO` or
                `logger.level == logging.DEBUG`
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors. Else, it is a list of
                pytorch tensors
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return. Overwrites any setting
                from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize(`bool`, *optional*, defaults to True):
                If set to true, returned vectors will have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity
                can be used
            tokenizer_kwargs(`Dict[str, Any]`, *optional*, defaults to {}):
                Keyword arguments for the tokenizer

        Returns:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked
            tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        _is_training = self.training
        self.eval()

        all_embeddings = []

        if not show_progress_bar:
            show_progress_bar = (
                logger.getEffectiveLevel() == INFO or logger.getEffectiveLevel() == DEBUG
            )
        if convert_to_tensor:
            convert_to_numpy = False

        _input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]  # pyright: ignore[reportAssignmentType]
            _input_was_string = True

        if device is not None:
            self.to(device)  # pyright: ignore[reportArgumentType]

        _permutation = np.argsort([-len(i) for i in sentences])
        _inverse_permutation = np.argsort(_permutation)
        sentences = [sentences[idx] for idx in _permutation]

        tokenizer_kwargs["padding"] = tokenizer_kwargs.get("padding", True)
        tokenizer_kwargs["max_length"] = tokenizer_kwargs.get("max_length", 512)
        tokenizer_kwargs["truncation"] = tokenizer_kwargs.get("truncation", True)

        if has_tqdm:
            assert trange is not None
            range_iter = trange(
                0,
                len(sentences),
                batch_size,
                desc="Encoding sentences",
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(sentences), batch_size)

        for i in range_iter:
            batch_sentences = sentences[i : i + batch_size]
            inputs = self.processor(
                text=batch_sentences,
                return_tensors="pt",
                add_instruction=True,
                **tokenizer_kwargs,
            ).to(self.device)
            embeddings = self.get_text_features(input_ids=inputs, normalize=normalize)
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.to(torch.float32).numpy() for emb in all_embeddings])
        if _input_was_string:
            all_embeddings = all_embeddings[0]

        self.train(_is_training)

        return all_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | BatchEncoding | BatchFeature | dict[str, torch.Tensor],
        pixel_values: torch.Tensor | BatchFeature,
        return_dict: bool | None = None,
        normalize: bool = True,
        return_loss: bool = False,
        *_,
        **__,
    ) -> tuple[torch.Tensor | None, ...] | ViCLIPOTOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_features = self.get_text_features(
            input_ids=input_ids, normalize=normalize, return_dict=return_dict
        )
        image_features = self.get_image_features(
            pixel_values=pixel_values, normalize=normalize, return_dict=return_dict
        )

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = logit_scale * (text_features @ image_features.t())
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            raise NotImplementedError("TODO")

        ouptuts = ViCLIPOTOutput(
            text_features=text_features,
            image_features=image_features,
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            logit_scale=logit_scale,
            logit_bias=self.logit_bias,
            loss=loss,
        )

        return ouptuts if return_dict else ouptuts.to_tuple()
