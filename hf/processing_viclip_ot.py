# pyright: reportIncompatibleMethodOverride=false
from typing import Literal, cast

import torch
import torchvision.transforms.v2 as v2
from transformers import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.models.clip import CLIPProcessor
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

InstructionMode = Literal["auto", "gemma", "e5", "qwen3", "bge", "sbert", "none"]

DEFAULT_QWEN3_INSTRUCTION = "Retrieve images or text relevant to the user's query."


def resolve_instruction_mode(model_name: str) -> InstructionMode:
    model_name = model_name.lower()
    if "gemma" in model_name:
        return "gemma"
    if "e5" in model_name:
        return "e5"
    if "qwen" in model_name:
        return "qwen3"
    if "bge" in model_name:
        return "bge"
    if "sbert" in model_name:
        return "sbert"
    raise ValueError(f"Unsupported model name for determining instruction mode: {model_name}")


def format_text_with_instruction(
    text: str | list[str],
    *,
    model_name: str | None = None,
    instruction_mode: InstructionMode = "auto",
    qwen_instruction: str = DEFAULT_QWEN3_INSTRUCTION,
) -> str | list[str]:
    if instruction_mode == "auto":
        if model_name is None:
            raise ValueError("`model_name` is required when instruction_mode='auto'.")
        instruction_mode = resolve_instruction_mode(model_name)

    if instruction_mode == "none":
        return text

    input_is_single = isinstance(text, str)
    texts = [text] if input_is_single else text

    if instruction_mode == "gemma":
        formatted = [f"sentence similarity | query: {sentence}" for sentence in texts]
    elif instruction_mode == "e5":
        formatted = [f"query: {sentence}" for sentence in texts]
    elif instruction_mode == "qwen3":
        formatted = [f"Instruct: {qwen_instruction}\nQuery:{sentence}" for sentence in texts]
    elif instruction_mode in ("bge", "sbert"):
        formatted = [f"{sentence}" for sentence in texts]
    else:
        raise ValueError(
            f"Invalid instruction_mode: {instruction_mode}. "
            "Expected one of ['auto', 'gemma', 'e5', 'qwen3', 'bge', 'sbert', 'none']"
        )

    return formatted[0] if input_is_single else formatted


class ViCLIPOTProcessor(CLIPProcessor):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        text_model_name: str | None = None,
        instruction_mode: InstructionMode = "auto",
        qwen_instruction: str = DEFAULT_QWEN3_INSTRUCTION,
        **kwargs,
    ) -> None:
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, **kwargs)
        self.text_model_name = text_model_name
        self.instruction_mode = instruction_mode
        self.qwen_instruction = qwen_instruction

    def format_text(self, text: str | list[str]) -> str | list[str]:
        return format_text_with_instruction(
            text,
            model_name=self.text_model_name,
            instruction_mode=cast(InstructionMode, self.instruction_mode),
            qwen_instruction=self.qwen_instruction,
        )

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = None,
        add_instruction: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if text is not None and add_instruction:
            text = self.format_text(text)  # pyright: ignore[reportArgumentType]

        return super().__call__(images=images, text=text, **kwargs)


class ViCLIPOTImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
    _valid_processor_keys = [
        "resize_size",
        "crop_size",
        "mean",
        "std",
        "interpolation",
    ]
    _interpolation_map = {
        "nearest": v2.InterpolationMode.NEAREST,
        "bilinear": v2.InterpolationMode.BILINEAR,
        "bicubic": v2.InterpolationMode.BICUBIC,
    }

    def __init__(
        self,
        resize_size: int | tuple[int, int] = 256,
        crop_size: int | tuple[int, int] = 224,
        mean: float | tuple[float, ...] = (0.485, 0.456, 0.406),
        std: float | tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: str = "bicubic",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.transform = self._build_transform()

    def _resolve_interpolation(self) -> v2.InterpolationMode:
        interpolation = str(self.interpolation).lower()
        if interpolation not in self._interpolation_map:
            raise ValueError(
                f"Unsupported interpolation mode: {self.interpolation}. "
                f"Expected one of {list(self._interpolation_map.keys())}."
            )
        return self._interpolation_map[interpolation]

    def _build_transform(self):
        return v2.Compose(
            [
                v2.Resize(self.resize_size, interpolation=self._resolve_interpolation()),
                v2.CenterCrop(size=self.crop_size),
                v2.ToTensor(),
                v2.Normalize(mean=self.mean, std=self.std),  # pyright: ignore[reportArgumentType]
            ]
        )

    def to_dict(self):
        output = super().to_dict()
        output.pop("transform", None)
        return output

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        transform_needs_rebuild = False
        for key, value in kwargs.items():
            if key in self._valid_processor_keys and value != getattr(self, key):
                setattr(self, key, value)
                transform_needs_rebuild = True

        if transform_needs_rebuild:
            self.transform = self._build_transform()

        images_list = make_list_of_images(images)
        output = torch.stack([self.transform(image) for image in images_list], dim=0)

        return BatchFeature(data={"pixel_values": output}, tensor_type=return_tensors)
