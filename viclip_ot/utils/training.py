import heapq
import os
import re
from contextlib import nullcontext
from typing import Any, NamedTuple, TypedDict

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data import DataLoader
from torchvision.ops.misc import FrozenBatchNorm2d
from tqdm.autonotebook import tqdm
from wandb.sdk.wandb_run import Run as WandbRun

from viclip_ot.utils.logger import logger
from viclip_ot.utils.metric import AverageMeter


class RandContext:
    """Snapshot/restore RNG so the GradCache second forward matches the first (dropout)."""

    def __init__(self, *tensors: Tensor) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


RECALL_METRIC_NAMES = (
    "i2t_R__1",
    "i2t_R__5",
    "i2t_R__10",
    "t2i_R__1",
    "t2i_R__5",
    "t2i_R__10",
)


class BootstrapCIResults(TypedDict):
    num_images: int
    num_resamples: int
    confidence_level: float
    seed: int
    intervals: dict[str, tuple[float, float]]


class RetrievalTensors(NamedTuple):
    logits_i2t: Tensor
    mask_i2t: Tensor
    logits_t2i: Tensor
    mask_t2i: Tensor
    unique_ids: np.ndarray
    caption_group_indices: np.ndarray


class EvalResults(TypedDict):
    loss: float

    i2t_mean_rank: float
    i2t_median_rank: float
    i2t_R__1: float
    i2t_R__5: float
    i2t_R__10: float

    t2i_mean_rank: float
    t2i_median_rank: float
    t2i_R__1: float
    t2i_R__5: float
    t2i_R__10: float

    alignment_score: float
    modality_gap: float


def get_parameter_names(model, forbidden_layer_types=None, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.

    Taken and modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L952

    Modifications:
    - Make `forbidden_layer_types` default to None
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names]
        if forbidden_layer_names is not None
        else []
    )
    if forbidden_layer_types is None:
        forbidden_layer_types = []

    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(
                pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns
            )
        ]
    # Add model specific parameters that are not in any child
    result += [
        k
        for k in model._parameters
        if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result


def convert_eval_results_to_dict(eval_results: EvalResults, fmt: str = "0.4f") -> dict[str, Any]:
    # TODO: ayo
    eval_results_dict: dict[str, Any] = {
        "loss": f"{eval_results['loss']:{fmt}}",
        "i2t_mean_rank": f"{eval_results['i2t_mean_rank']:{fmt}}",
        "i2t_median_rank": f"{eval_results['i2t_median_rank']:{fmt}}",
        "i2t_R__1": f"{eval_results['i2t_R__1']:{fmt}}",
        "i2t_R__5": f"{eval_results['i2t_R__5']:{fmt}}",
        "i2t_R__10": f"{eval_results['i2t_R__10']:{fmt}}",
        "t2i_mean_rank": f"{eval_results['t2i_mean_rank']:{fmt}}",
        "t2i_median_rank": f"{eval_results['t2i_median_rank']:{fmt}}",
        "t2i_R__1": f"{eval_results['t2i_R__1']:{fmt}}",
        "t2i_R__5": f"{eval_results['t2i_R__5']:{fmt}}",
        "t2i_R__10": f"{eval_results['t2i_R__10']:{fmt}}",
        "alignment_score": f"{eval_results['alignment_score']:{fmt}}",
        "modality_gap": f"{eval_results['modality_gap']:{fmt}}",
    }
    return eval_results_dict


class EarlyStopping:
    def __init__(
        self, patience: int = 5, min_delta: float = 0.0, enabled: bool = True, verbose: bool = True
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.counter = 0
        self.min_val_loss = float("inf")
        self.verbose = verbose

    def early_stop(self, val_loss: float) -> bool:
        if not self.enabled:
            # do nothing
            return False

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"No improvement in validation loss for {self.counter} consecutive epochs"
                )
            if self.counter >= self.patience:
                return True
        return False

    def is_enabled(self) -> bool:
        return self.enabled


def make_model(
    model_name: str, num_classes: int, pretrained: bool = True, linear_probing: bool = False
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet50":
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        classifier = model.fc
    elif model_name == "densenet121":
        model = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None,
        )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        classifier = model.classifier
    elif model_name.startswith("timm/"):
        model = timm.create_model(
            model_name[len("timm/") :],
            pretrained=pretrained,
            num_classes=num_classes,
        )
        classifier_str = model.default_cfg.get("classifier")  # pyright: ignore
        classifier = None
        if classifier_str is not None:
            classifier = get_submodule_from_module_name(model=model, module_path=classifier_str)
        if classifier is None:
            logger.warning(
                "Failed to get classifier from timm model default_cfg, fall back to inferring classifier manually"
            )
            classifier = infer_final_fc(model)
        assert classifier is not None and isinstance(classifier, nn.Linear)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if linear_probing:
        # freeze all layers except the final classifier layer
        for param in model.parameters():
            param.requires_grad = False
        for param in classifier.parameters():
            param.requires_grad = True

    # initializing classification head
    nn.init.xavier_uniform_(classifier.weight)
    if classifier.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
        nn.init.zeros_(classifier.bias)

    return model


def infer_final_fc(model: nn.Module) -> nn.Module:
    if isinstance(model, torchvision.models.ResNet):
        final_fc = model.fc
    elif isinstance(model, torchvision.models.DenseNet):
        final_fc = model.classifier
    else:
        # timm models
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            final_fc = model.head
        elif (
            hasattr(model, "head")
            and hasattr(model.head, "fc")
            and isinstance(model.head, nn.Module)
            and isinstance(model.head.fc, nn.Linear)
        ):
            final_fc = model.head.fc
        elif (
            hasattr(model, "head")
            and hasattr(model.head, "fc")
            and isinstance(model.head, nn.Module)
            and isinstance(model.head.fc, timm.models.metaformer.MlpHead)
        ):
            final_fc = model.head.fc.fc2
        else:
            raise ValueError("Unsupported model type for inferring final fc layer")

    return final_fc


def eval_model(
    model: nn.Module,
    criterion,
    eval_data_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    device: torch.device,
    autocast_context=None,
    bootstrap_num_resamples: int = 0,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    bootstrap_batch_size: int = 256,
) -> EvalResults:
    if autocast_context is None:
        autocast_context = nullcontext()
    if bootstrap_num_resamples < 0:
        raise ValueError("bootstrap_num_resamples must be non-negative")
    if not 0.0 < bootstrap_confidence_level < 1.0:
        raise ValueError("bootstrap_confidence_level must be between 0 and 1")
    if bootstrap_batch_size < 1:
        raise ValueError("bootstrap_batch_size must be at least 1")

    model_mode_before = model.training
    model.eval()

    eval_iter = tqdm(eval_data_loader, desc="Evaluating model")
    eval_loss = AverageMeter("eval_loss", fmt=":0.4f")

    all_image_features = []
    all_text_features = []
    all_image_ids = []
    logit_scale = None
    with torch.inference_mode():
        for batch in eval_iter:
            images = batch["images"].to(device=device, non_blocking=True)
            text_inputs = batch["text_inputs"].to(device=device, non_blocking=True)
            image_ids = batch["image_ids"]

            with autocast_context:
                model_outputs = model(images, text_inputs)
                image_features = model_outputs["image_features"]
                text_features = model_outputs["text_features"]
                logit_scale = model_outputs["logit_scale"]

                loss = criterion(
                    image_features=image_features,
                    text_features=text_features,
                    logit_scale=model_outputs["logit_scale"],
                    logit_bias=model_outputs.get("logit_bias", None),
                    image_ids=image_ids,
                )

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            all_image_ids.append(image_ids)

            eval_loss.update(loss.item(), images.shape[0])

            eval_iter.set_postfix(
                {
                    "loss": f"{loss.item():0.4f}",
                }
            )

        assert logit_scale is not None
        retrieval_tensors = _build_retrieval_tensors(
            image_features=torch.cat(all_image_features, dim=0),
            text_features=torch.cat(all_text_features, dim=0),
            logit_scale=logit_scale.cpu(),
            image_ids=torch.cat(all_image_ids, dim=0),
        )
        eval_metrics = _get_retrieval_metrics_from_tensors(retrieval_tensors)
        eval_metrics["loss"] = eval_loss.avg

        if bootstrap_num_resamples > 0:
            eval_metrics["bootstrap_ci"] = _bootstrap_retrieval_confidence_intervals_from_tensors(
                retrieval_tensors=retrieval_tensors,
                num_resamples=bootstrap_num_resamples,
                confidence_level=bootstrap_confidence_level,
                seed=bootstrap_seed,
                batch_size=bootstrap_batch_size,
            )
        alignment_metrics = compute_alignment_score(
            image_embeddings=torch.cat(all_image_features, dim=0).numpy(),
            text_embeddings=torch.cat(all_text_features, dim=0).numpy(),
        )
        eval_metrics.update(alignment_metrics)
    # set model back to the original mode
    model.train(model_mode_before)

    return eval_metrics  # pyright: ignore[reportReturnType]


def _build_retrieval_tensors(
    image_features: Tensor,
    text_features: Tensor,
    logit_scale: Tensor,
    image_ids: Tensor,
) -> RetrievalTensors:
    """Build the shared retrieval logits, masks, and image-caption groups."""
    image_features = image_features.detach().cpu().float()
    text_features = text_features.detach().cpu().float()
    image_ids = image_ids.detach().cpu()
    logit_scale = logit_scale.detach().cpu().float()

    unique_ids, first_indices = np.unique(image_ids.numpy(), return_index=True)
    unique_image_features = image_features[torch.from_numpy(first_indices)]

    logits_i2t = logit_scale * unique_image_features @ text_features.t()
    mask_i2t = torch.from_numpy(unique_ids[:, None] == image_ids.numpy()[None, :])

    logits_t2i = logit_scale * text_features @ unique_image_features.t()
    mask_t2i = mask_i2t.t()

    caption_group_indices = np.searchsorted(unique_ids, image_ids.numpy())
    return RetrievalTensors(
        logits_i2t=logits_i2t,
        mask_i2t=mask_i2t,
        logits_t2i=logits_t2i,
        mask_t2i=mask_t2i,
        unique_ids=unique_ids,
        caption_group_indices=caption_group_indices,
    )


def _bootstrap_retrieval_confidence_intervals_from_tensors(
    retrieval_tensors: RetrievalTensors,
    num_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
    batch_size: int = 256,
) -> BootstrapCIResults:
    """Estimate bootstrap intervals from precomputed retrieval tensors."""
    num_images = len(retrieval_tensors.unique_ids)
    if num_images == 0:
        raise ValueError("Cannot bootstrap an empty retrieval dataset")

    # Compute per-query hits once. The bootstrap then only resamples these
    # fixed outcomes, not model features or rankings.
    i2t_hits = _compute_retrieval_hits(
        retrieval_tensors.logits_i2t,
        retrieval_tensors.mask_i2t,
    ).numpy()
    t2i_hits = _compute_retrieval_hits(
        retrieval_tensors.logits_t2i,
        retrieval_tensors.mask_t2i,
    ).numpy()

    caption_counts = np.bincount(
        retrieval_tensors.caption_group_indices,
        minlength=num_images,
    )
    if np.any(caption_counts == 0):
        raise ValueError("Every image must have at least one caption")
    t2i_hits_by_image = np.stack(
        [
            np.bincount(
                retrieval_tensors.caption_group_indices,
                weights=t2i_hits[:, column],
                minlength=num_images,
            )
            for column in range(t2i_hits.shape[1])
        ],
        axis=1,
    )

    rng = np.random.default_rng(seed)
    metric_samples = np.empty((num_resamples, len(RECALL_METRIC_NAMES)), dtype=np.float64)
    sampling_probabilities = np.full(num_images, 1.0 / num_images)

    # Process bootstrap draws in batches to limit temporary memory use. Each
    # row draws `num_images` IDs with replacement.
    for start in range(0, num_resamples, batch_size):
        stop = min(start + batch_size, num_resamples)
        batch_image_counts = rng.multinomial(
            num_images,
            sampling_probabilities,
            size=stop - start,
        )
        # Counts preserve duplicate image clusters exactly.

        sampled_i2t_denominator = float(num_images)
        sampled_t2i_denominator = batch_image_counts @ caption_counts
        metric_samples[start:stop, 0:3] = (batch_image_counts @ i2t_hits) / sampled_i2t_denominator
        metric_samples[start:stop, 3:6] = (
            batch_image_counts @ t2i_hits_by_image
        ) / sampled_t2i_denominator[:, None]

    alpha = (1.0 - confidence_level) / 2.0
    quantiles = np.quantile(
        metric_samples,
        [alpha, 1.0 - alpha],
        axis=0,
        method="linear",
    )
    intervals = {
        metric_name: (float(quantiles[0, i]), float(quantiles[1, i]))
        for i, metric_name in enumerate(RECALL_METRIC_NAMES)
    }
    # Compute the interval for the reported six-way average from each
    # replicate, rather than averaging six separately computed bounds.
    average_samples = metric_samples.mean(axis=1)
    average_quantiles = np.quantile(
        average_samples,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    intervals["average_R"] = (float(average_quantiles[0]), float(average_quantiles[1]))
    return {
        "num_images": num_images,
        "num_resamples": num_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "intervals": intervals,
    }


def bootstrap_retrieval_confidence_intervals(
    image_features: Tensor,
    text_features: Tensor,
    logit_scale: Tensor,
    image_ids: Tensor,
    num_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
    batch_size: int = 256,
) -> BootstrapCIResults:
    """Estimate retrieval-metric confidence intervals with an image-cluster bootstrap.

    Each bootstrap replicate samples image queries with replacement and keeps the
    complete, original gallery fixed. All captions belonging to a sampled image
    are included as text queries, so image-caption dependence is preserved.
    """
    if num_resamples < 1:
        raise ValueError("num_resamples must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("image_features and text_features must be rank-2 tensors")
    if image_features.shape[0] != image_ids.shape[0]:
        raise ValueError("image_features and image_ids must have the same number of rows")
    if text_features.shape[0] != image_ids.shape[0]:
        raise ValueError("text_features and image_ids must have the same number of rows")
    if image_ids.ndim != 1:
        raise ValueError("image_ids must be a rank-1 tensor")

    retrieval_tensors = _build_retrieval_tensors(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        image_ids=image_ids,
    )
    return _bootstrap_retrieval_confidence_intervals_from_tensors(
        retrieval_tensors=retrieval_tensors,
        num_resamples=num_resamples,
        confidence_level=confidence_level,
        seed=seed,
        batch_size=batch_size,
    )


def _compute_retrieval_hits(logits: Tensor, mask: Tensor, k_vals=(1, 5, 10)) -> Tensor:
    """Return per-query Recall@K hits with shape ``[queries, len(k_vals)]``."""
    max_k = min(max(k_vals), logits.shape[1])
    _, top_indices = logits.topk(max_k, dim=1)  # [B, max_k]

    # gather ground truth booleans at the retrieved positions
    rows = torch.arange(logits.shape[0]).view(-1, 1)
    retrieved_mask = mask[rows, top_indices]  # [B, max_k]

    return torch.stack([retrieved_mask[:, : min(k, max_k)].any(dim=1) for k in k_vals], dim=1)


def _get_retrieval_metrics_from_tensors(
    retrieval_tensors: RetrievalTensors,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics.update(
        _compute_retrieval_metrics(
            retrieval_tensors.logits_i2t,
            retrieval_tensors.mask_i2t,
            prefix="i2t",
        )
    )
    metrics.update(
        _compute_retrieval_metrics(
            retrieval_tensors.logits_t2i,
            retrieval_tensors.mask_t2i,
            prefix="t2i",
        )
    )
    return metrics


def get_retrieval_metrics(
    image_features: Tensor,
    text_features: Tensor,
    logit_scale: Tensor,
    image_ids: Tensor | None = None,
) -> dict[str, Any]:
    """
    If `image_ids` is provided, the computation will take into account image with multiple captions.
    """
    if image_ids is None:
        # 1-1 image caption mapping
        image_ids = torch.arange(len(image_features))

    retrieval_tensors = _build_retrieval_tensors(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        image_ids=image_ids,
    )

    # Image-to-Text
    # Query:   Unique Images [N_unique]
    # Gallery: All Texts     [N_total]
    # Protocol: For each unique image, did we find ANY of its captions?
    # Logits: [N_unique, N_total]
    # Text-to-Image
    # Query:   All Texts     [N_total]
    # Gallery: Unique Images [N_unique]
    # Protocol: For each caption, did we find the ONE correct image?
    # Logits: [N_total, N_unique]
    return _get_retrieval_metrics_from_tensors(retrieval_tensors)


def _compute_retrieval_metrics(
    logits: Tensor, mask: Tensor, prefix: str, k_vals=(1, 5, 10)
) -> dict[str, Any]:
    """ "Compute recall@k and mean rank."""

    results = {}
    # gather ground truth booleans at the retrieved positions
    hits_by_k = _compute_retrieval_hits(logits, mask, k_vals)

    for k_idx, k in enumerate(k_vals):
        # hit if at least one of the top k is True
        hits = hits_by_k[:, k_idx]
        results[f"{prefix}_R__{k}"] = hits.float().mean().item()

    argsort = torch.argsort(logits, dim=1, descending=True)

    # sorted_mask[i, j] is True if the item at rank 'j' is a match
    rows = torch.arange(logits.shape[0]).view(-1, 1)
    sorted_mask = mask[rows, argsort]

    # find the first rank (min index) where sorted_mask is True
    rank_matrix = torch.arange(logits.shape[1]).view(1, -1).float()
    masked_ranks = rank_matrix.expand(logits.shape[0], -1).clone()
    masked_ranks[~sorted_mask] = float("inf")

    # get the "Best Rank" (lowest index) for every row
    best_rank_per_row = masked_ranks.min(dim=1).values

    # convert 0-indexed to 1-indexed
    best_rank_per_row = best_rank_per_row.numpy() + 1

    results[f"{prefix}_mean_rank"] = np.mean(best_rank_per_row)
    results[f"{prefix}_median_rank"] = np.floor(np.median(best_rank_per_row))

    return results


def save_top_k_checkpoints(
    criterion_metrics: list[str],
    top_k: int,
    val_results: EvalResults,
    best_val_results: dict[str, list[tuple[float, str]]],
    state_dict_to_save: dict[str, Any],
    checkpoint_path_template: str,
) -> None:
    """
    Save the top-k model checkpoints based on validation metrics.

    This function implements a checkpoint management strategy that
    maintains only the top-k best checkpoints for each specified metric.
    It uses a min-heap data structure to efficiently track and manage
    the best performing models, automatically removing worse checkpoints
    when the limit is exceeded.

    Behavior:
        - For each metric in `criterion_metrics`, evaluates if the current model performance
          warrants saving a checkpoint
        - For loss metrics, uses negative values to maintain consistent "higher is better" semantics
        - Maintains up to `top_k` checkpoints per metric using a min-heap
        - When the checkpoint limit is reached, replaces the worst checkpoint if current performance is better
        - Automatically deletes old checkpoint files from disk when they are replaced
        - Saves complete checkpoint state including model weights, validation results, epoch, and global step

    File Naming:
        Checkpoint files are named as: "model_1_epoch_{epoch}_{metric}_{metric_value:.4f}.pth"

    Note:
        - The function modifies best_val_results in-place to maintain state across training epochs
        - Loss values are negated internally for heap operations but displayed as positive values in logs
        - Only saves checkpoints that are among the top-k performing for their respective metrics
    """
    for metric in criterion_metrics:
        if metric.startswith("_"):
            # for metric with prefix '_', we want to save the lowest value
            metric = metric[1:]
            current_metric_value = -val_results[metric]
        else:
            current_metric_value = val_results[metric]

        current_checkpoint_path = checkpoint_path_template.format(
            metric=metric, metric_value=abs(current_metric_value)
        )

        if len(best_val_results[metric]) < top_k:
            # If we haven't saved `top_k`` checkpoints yet, just add this one.
            # Store the actual positive metric value.
            heapq.heappush(
                best_val_results[metric],
                (current_metric_value, current_checkpoint_path),
            )

            # Save the model state and other relevant information
            torch.save(
                state_dict_to_save,
                current_checkpoint_path,
            )
            logger.info(
                f"Saved checkpoint for {metric}: {abs(current_metric_value):.6f} to {current_checkpoint_path}"
            )
        else:
            # If we already have args.save_best_k checkpoints, check if the current one is better than the worst of them.
            # The worst of the k is at the top of the min-heap (best_val_results[metric][0]).
            worst_of_k_value = best_val_results[metric][0][
                0
            ]  # This correctly gets the smallest (worst) value in the heap

            if current_metric_value > worst_of_k_value:
                # Current checkpoint is better, so replace the worst one in the heap
                # heapq.heapreplace pops the smallest item and then pushes the new item
                old_worst_checkpoint_tuple = heapq.heapreplace(
                    best_val_results[metric],
                    (current_metric_value, current_checkpoint_path),
                )
                old_worst_path = old_worst_checkpoint_tuple[1]

                # Delete the old worst checkpoint file from disk
                if os.path.exists(old_worst_path):
                    os.remove(old_worst_path)
                    print(f"Deleted old worst checkpoint: {old_worst_path}")

                # Save the new better checkpoint
                torch.save(
                    state_dict_to_save,
                    current_checkpoint_path,
                )
                logger.info(
                    f"Replaced checkpoint for {metric}: {abs(current_metric_value):.6f} "
                    f"(old worst: {abs(worst_of_k_value):.6f}) to {current_checkpoint_path}",
                )


def maybe_log_eval_results(
    eval_results: EvalResults,
    epoch: int,
    prefix: str = "val",
    class_names: list[str] | None = None,
    wandb_run: WandbRun | None = None,
    wandb_log_step: int | None = None,
) -> None:
    """
    Log evaluation results to Weights & Biases (wandb) if a wandb_run is provided.
    """
    if wandb_run is None:
        return

    assert wandb_log_step is not None

    log_data = {
        f"{prefix}/loss": eval_results["loss"],
        f"{prefix}/i2t_mean_rank": eval_results["i2t_mean_rank"],
        f"{prefix}/i2t_median_rank": eval_results["i2t_median_rank"],
        f"{prefix}/i2t_R__1": eval_results["i2t_R__1"],
        f"{prefix}/i2t_R__5": eval_results["i2t_R__5"],
        f"{prefix}/i2t_R__10": eval_results["i2t_R__10"],
        f"{prefix}/t2i_mean_rank": eval_results["t2i_mean_rank"],
        f"{prefix}/t2i_median_rank": eval_results["t2i_median_rank"],
        f"{prefix}/t2i_R__1": eval_results["t2i_R__1"],
        f"{prefix}/t2i_R__5": eval_results["t2i_R__5"],
        f"{prefix}/t2i_R__10": eval_results["t2i_R__10"],
        f"{prefix}/alignment_score": eval_results["alignment_score"],
        f"{prefix}/modality_gap": eval_results["modality_gap"],
        "epoch": epoch + 1,
    }

    wandb_run.log(log_data, step=wandb_log_step)


def print_eval_results(
    eval_results: EvalResults,
    prefix: str,  # either 'val' or 'test'
    epoch: int | None = None,
    logger=None,
) -> None:
    print_prefix = ""
    if epoch is None:
        print_prefix = f"{prefix} results: "
    else:
        print_prefix = f"{prefix} results on epoch {epoch}: "

    print_str = (
        f"{print_prefix}"
        f"{prefix}_loss {eval_results['loss']:0.4f} | "
        f"{prefix}_i2t_mean_rank {eval_results['i2t_mean_rank']:0.4f} | "
        f"{prefix}_i2t_median_rank {eval_results['i2t_median_rank']:0.4f} | "
        f"{prefix}_i2t_R@1 {eval_results['i2t_R__1']:0.4f} | "
        f"{prefix}_i2t_R@5 {eval_results['i2t_R__5']:0.4f} | "
        f"{prefix}_i2t_R@10 {eval_results['i2t_R__10']:0.4f} | "
        f"{prefix}_t2i_mean_rank {eval_results['t2i_mean_rank']:0.4f} | "
        f"{prefix}_t2i_median_rank {eval_results['t2i_median_rank']:0.4f} | "
        f"{prefix}_t2i_R@1 {eval_results['t2i_R__1']:0.4f} | "
        f"{prefix}_t2i_R@5 {eval_results['t2i_R__5']:0.4f} | "
        f"{prefix}_t2i_R@10 {eval_results['t2i_R__10']:0.4f} | "
        f"{prefix}_alignment_score {eval_results['alignment_score']:0.4f} | "
        f"{prefix}_modality_gap {eval_results['modality_gap']:0.4f}"
    )
    print(print_str)
    if logger is not None:
        logger.info(print_str)


def accuracy(output, target, topk=(1,)) -> list[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum().item() / batch_size for k in topk]


def compute_alignment_score(
    image_embeddings: np.ndarray, text_embeddings: np.ndarray
) -> dict[str, Any]:
    assert image_embeddings.shape == text_embeddings.shape, "Shapes must match"

    # alignment score = average cosine similarity of matched image-text pairs
    alignment_score = np.sum(image_embeddings * text_embeddings) / image_embeddings.shape[0]

    # modality gap = difference between centroids of image and text embeddings
    image_centroid = np.mean(image_embeddings, axis=0)
    text_centroid = np.mean(text_embeddings, axis=0)

    modality_gap = np.linalg.norm(text_centroid - image_centroid)

    return {
        "alignment_score": alignment_score,
        "modality_gap": modality_gap,
    }


def get_submodule_from_module_name(
    model: torch.nn.Module, module_path: str
) -> torch.nn.Module | None:
    """Get a submodule by dot-separated path, e.g. "head.fc"."""
    parts = module_path.split(".")
    sub = model
    for p in parts:
        try:
            sub = getattr(sub, p)
        except AttributeError:
            return None
    return sub


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
