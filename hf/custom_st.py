# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false
"""
Adopted from: https://huggingface.co/jinaai/jina-clip-v2/blob/main/custom_st.py
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Literal

import requests
import torch
from PIL import Image
from sentence_transformers.models.InputModule import InputModule
from transformers import AutoConfig, AutoModel, AutoProcessor


class ViCLIPOT(InputModule):
    save_in_root: bool = True

    def __init__(
        self,
        model_name_or_path: str = "minhnguyent546/ViCLIP-OT",
        processor_name_or_path: str | None = None,
        max_seq_length: int | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        assume_text_inputs: bool = False,
        cache_dir: str | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        **kwargs,
    ) -> None:
        """
        Creates a custom SentenceTransformer module that uses `minhnguyent546/ViCLIP-OT` to
        map sentences/images to embeddings
        Args:
            model_name_or_path (str, optional): If it is a filepath on disc, it loads
                the model from that path. If it is not a path, tries to construct a
                model from the Hugging Face Hub with that name. Defaults to
                'minhnguyent546/ViCLIP-OT'
            processor_name_or_path (str, optional): If it is a filepath on disc, it
                loads the processor from that path. If it is not a path, tries to
                construct a processor from the Hugging Face Hub with that name.
                If `None` it is automatically set to the value of `model_name_or_path`
            max_seq_length (int, optional): The maximum sequence length of the model.
                If not provided, will be inferred from model or tokenizer
            config_kwargs (Dict[str, Any], optional): Additional model configuration
                parameters to be passed to the Hugging Face Transformers config
            model_kwargs (Dict[str, Any], optional): Additional model configuration
                parameters to be passed to the Hugging Face Transformers model
            tokenizer_kwargs (Dict[str, Any], optional): Additional processor configuration
                parameters to be passed to the Hugging Face Transformers processor
            assume_text_inputs (bool, optional): If set to `True`, all inputs are
                treated as texts. Defaults to `False`
            cache_dir (str, optional): The Hugging Face Hub cache directory
            backend (str, optional): Computational backend, only 'torch' is supported
        """
        super().__init__()
        if backend != "torch":
            raise ValueError(f"Backend '{backend}' is not supported, please use 'torch' instead")

        config_kwargs = config_kwargs or {}
        model_kwargs = model_kwargs or {}
        processor_kwargs = tokenizer_kwargs or {}

        common_kwargs = {
            "token": kwargs.get("token", None),
            "trust_remote_code": kwargs.get("trust_remote_code", False),
            "revision": kwargs.get("revision", None),
            "local_files_only": kwargs.get("local_files_only", None),
        }

        for kwargs_instance in (model_kwargs, processor_kwargs, config_kwargs):
            for common_key in ("token", "trust_remote_code", "revision", "local_files_only"):
                if common_key not in kwargs_instance and common_key in common_kwargs:
                    kwargs_instance[common_key] = common_kwargs[common_key]

        config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **config_kwargs
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_kwargs
        )
        if max_seq_length is not None and "model_max_length" not in processor_kwargs:
            processor_kwargs["model_max_length"] = max_seq_length

        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path or model_name_or_path,
            cache_dir=cache_dir,
            **processor_kwargs,
        )
        self.assume_text_inputs = assume_text_inputs

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            _MAX_SEQ_LENGTH = 2**63 - 1
            max_seq_length = _MAX_SEQ_LENGTH
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "max_position_embeddings"
            ):
                max_seq_length = min(max_seq_length, self.model.config.max_position_embeddings)
            if hasattr(self.processor, "tokenizer") and hasattr(
                self.processor.tokenizer, "model_max_length"
            ):
                max_seq_length = min(max_seq_length, self.processor.tokenizer.model_max_length)

            if max_seq_length == _MAX_SEQ_LENGTH:
                max_seq_length = None

        self.max_seq_length = max_seq_length

    def __repr__(self) -> str:
        return "ViCLIPOTModel()"

    @property
    def tokenizer(self):
        return self.processor

    @staticmethod
    def _decode_data_image(data_image_str: str) -> Image.Image:
        _header, data = data_image_str.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))

    def tokenize(
        self, texts: list[str | Image.Image], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Encodes input samples. Text samples are tokenized. Image URLs, image data
        buffers and PIL images are passed through the image processor.
        """
        _pil_images = []
        _texts = []
        _image_text_info = []  # 0 for image, 1 for text

        if self.assume_text_inputs:
            for sample in texts:
                if isinstance(sample, str):
                    _texts.append(sample)
                    _image_text_info.append(1)
        else:
            for sample in texts:
                pil_image = None
                if isinstance(sample, str):
                    if sample.startswith("http"):
                        try:
                            response = requests.get(sample)
                            pil_image = Image.open(BytesIO(response.content))
                            _image_text_info.append(0)
                        except Exception as e:
                            _ = str(e)
                            _texts.append(sample)
                            _image_text_info.append(1)
                    elif sample.startswith("data:image/"):
                        pil_image = self._decode_data_image(sample)
                        _image_text_info.append(0)
                    else:
                        try:
                            pil_image = Image.open(sample)
                            _image_text_info.append(0)
                        except Exception as e:
                            _ = str(e)
                            _texts.append(sample)
                            _image_text_info.append(1)
                elif isinstance(sample, Image.Image):  # pyright: ignore
                    pil_image = sample
                    _image_text_info.append(0)

                if pil_image is not None:
                    # handle palette images with transparency
                    if pil_image.mode == "P" and "transparency" in pil_image.info:
                        pil_image = pil_image.convert("RGBA")

                    pil_image = pil_image.convert("RGB")

                    _pil_images.append(pil_image)

        encoding = {}
        if len(_texts):
            encoding = self.processor(
                text=_texts,
                padding=padding,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_length,
                add_instruction=True,
            )

        if len(_pil_images):
            encoding["pixel_values"] = self.processor(
                images=_pil_images, return_tensors="pt"
            ).pixel_values

        encoding["image_text_info"] = _image_text_info
        return dict(encoding)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_embeddings = []
        text_embeddings = []

        if "pixel_values" in features:
            image_embeddings = self.model.get_image_features(
                features["pixel_values"], normalize=False
            )
        if "input_ids" in features:
            input_ids = {
                "input_ids": features["input_ids"],
                "attention_mask": features.get("attention_mask", None),
                "token_type_ids": features.get("token_type_ids", None),
            }

            text_embeddings = self.model.get_text_features(
                input_ids=input_ids,
                normalize=False,
            )

        sentence_embedding = []
        image_features = iter(image_embeddings)
        text_features = iter(text_embeddings)
        for _, _input_type in enumerate(features["image_text_info"]):
            if _input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))

        features["sentence_embedding"] = torch.stack(sentence_embedding).float()
        return features

    def save(self, output_path: str, safe_serialization: bool = True, **kwargs) -> None:
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.processor.save_pretrained(output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        # Loading-specific keyword arguments
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        # Module-specific keyword arguments
        # trust_remote_code,
        # model_kwargs,
        # tokenizer_kwargs,
        # config_kwargs,
        # backend,
        **kwargs,
    ) -> ViCLIPOT:
        local_path = cls.load_dir_path(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(
            local_path, token=token, revision=revision, local_files_only=local_files_only, **kwargs
        )
