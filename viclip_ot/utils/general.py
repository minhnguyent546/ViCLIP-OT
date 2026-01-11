import argparse
import os
import random
from typing import Any

import fvcore.nn
import numpy as np
import torch
import torch.nn as nn
import yaml
from wandb.sdk.wandb_run import Run as WandbRun


def save_metadata_to_checkpoint(
    checkpoint_dir: str,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    wandb_run: WandbRun | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    metadata = {
        "args": vars(args),
        "model_config": model_config,
    }

    if wandb_run is not None:
        metadata["wandb"] = {
            "id": wandb_run.id,
            "name": wandb_run.name,
            "project": wandb_run.project,
            "tags": wandb_run.tags,
            "notes": wandb_run.notes,
        }

    metadata_path = os.path.join(checkpoint_dir, "metadata.yml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_tensor_or_np(obj: Any) -> bool:
    return isinstance(obj, (torch.Tensor, np.ndarray))


def get_batch_samples(
    data_iter,
    num_batches: int,
    labels_key: str | None = None,
    labels_index: int | None = None,
    ignore_index: int = -100,
) -> tuple[list[Any], int | None]:
    """
    Get a batch of samples from the data iterator.

    Note: this function is applied for data iterators that produce items
    that have labels with labels key `labels_key` if items are
    of type dict, or labels index `labels_index` if items are
    of type list or tuple.
    """
    if labels_key is not None and labels_index is not None:
        raise ValueError("Specify either 'labels_key' or 'labels_index', not both.")

    batch_samples = []
    num_items_in_batch = None
    for _ in range(num_batches):
        try:
            batch_samples.append(next(data_iter))
        except StopIteration:
            break

    # there are no samples left
    if not batch_samples:
        return batch_samples, None

    # TODO: for simplicity, we handle for Tensor only here and
    # do not support list or numpy array
    labels_id = labels_key or labels_index
    if not isinstance(batch_samples[0][labels_id], torch.Tensor):
        raise ValueError(
            f"Expected labels in batch samples to be of type torch.Tensor, found type {type(batch_samples[0][labels_id])}"
        )

    if batch_samples[0][labels_id].ndim == 1:
        num_items_in_batch = sum(
            torch.count_nonzero(batch_sample[labels_id] != ignore_index).item()
            for batch_sample in batch_samples
        )
    else:
        num_items_in_batch = sum(
            batch_sample[labels_id].shape[0] for batch_sample in batch_samples
        )
    return batch_samples, num_items_in_batch  # pyright: ignore[reportReturnType]


def is_checkpoint_file(f: str) -> bool:
    return os.path.isfile(f) and f.endswith(".pth")


def find_checkpoint_files(checkpoint_files_or_dirs: list[str]) -> list[str]:
    """Get all checkpoints files from the given list of files or directories."""
    checkpoint_paths: list[str] = []

    for checkpoint_path in checkpoint_files_or_dirs:
        if is_checkpoint_file(checkpoint_path):
            checkpoint_paths.append(checkpoint_path)
        elif os.path.isdir(checkpoint_path):
            checkpoint_paths.extend(
                os.path.join(checkpoint_path, f)
                for f in os.listdir(checkpoint_path)
                if is_checkpoint_file(os.path.join(checkpoint_path, f))
            )
    # remove duplicates
    checkpoint_paths = list(set(checkpoint_paths))
    return checkpoint_paths


def get_device(device_str: str | None = None) -> torch.device:
    if device_str is None or device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            try:
                _ = torch.ones(1, device="mps")
                device_str = "mps"
            except Exception:
                device_str = "cpu"
        else:
            device_str = "cpu"

    try:
        return torch.device(device_str)
    except Exception as e:
        raise ValueError(f"Invalid device specification: {device_str}") from e


def to_hms(seconds: float) -> str:
    """Convert seconds to hours, minutes, seconds format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"


def to_human_readable(num, format="0.2f") -> str:
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1e3:
            return f"{num:{format}}{unit}"
        num /= 1.0e3

    return f"{num:{format}}E"


def count_model_params(model: nn.Module, trainable: bool = False) -> int:
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_model_flops(model: nn.Module, input_size: tuple[int, ...]) -> int:
    flops = fvcore.nn.FlopCountAnalysis(model, torch.randn(*input_size))
    return flops.total()


def load_yaml_file(file_path: str) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return data
