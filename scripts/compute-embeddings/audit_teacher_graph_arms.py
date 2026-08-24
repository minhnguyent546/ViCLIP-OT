"""
Post-training audit of one SIGROT arm checkpoint (zero training cost).

For a given checkpoint + the teacher embedding files used by that arm, this script
reports, at the checkpoint's ACTUAL learned logit_scale:

1. Standard test-split retrieval metrics (independent re-evaluation).
2. Exact random-ranking chance levels for the test split (analytic R@K and mean rank).
3. Train-gallery retrieval (memorization vs transfer check).
4. Per-update graph prior and transport-plan diagnostics on fixed train batches.
   With GradCache, each audited plan spans `gradient_accum_steps` loader batches.
   It is a pair-level square of size (all flattened captions in that update),
   matching training exactly, not image-count-sized.
5. Student-graph recovery on a subsample of PAIRS: off-diagonal Pearson correlation
   between student and teacher combined similarity matrices, top-k neighbor overlap,
   and effective rank (participation ratio) of student features.

Feature-row alignment uses `pair_ids` emitted by ImageTextCollate, or derives them
from the emitted replacement indices on older dataset code. This matches training when
ImageTextDataset skips an unreadable image and substitutes the next readable sample.

Example:
    uv run --no-sync python scripts/compute-embeddings/audit_teacher_graph_arms.py \
        --checkpoint ./checkpoints/sigrot_random/model_epoch_XX_t2i_R__1_0.XXXX.pth \
        --arm_label random \
        --teacher_image_embeddings ./data/UIT-OpenViIC-embeddings/train_image_embeddings_qwen3_vl_embedding_2b_random.pt \
        --teacher_caption_embeddings ./data/UIT-OpenViIC-embeddings/train_caption_embeddings_qwen3_vl_embedding_2b_random.pt \
        --output_json ./audit_results/random.json
"""

import argparse
import json
import math
import os
from typing import NamedTuple

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch import Tensor
from torch.utils.data import DataLoader

import viclip_ot.constants as C
import viclip_ot.losses as losses
import viclip_ot.utils as utils
from viclip_ot.dataset import ImageTextCollate, ImageTextDataset
from viclip_ot.model import ViCLIPOT, ViCLIPOTConfig
from viclip_ot.utils.logger import init_logger, logger
from viclip_ot.utils.training import get_retrieval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one SIGROT arm checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm_label", required=True, help="e.g. clean, noise_rho0.5, random")
    parser.add_argument("--model_config", default="./config/model.vit_base_dinov3_sbert.yaml")
    parser.add_argument("--dataset_dir", default="./data/UIT-OpenViIC")
    parser.add_argument("--teacher_image_embeddings", required=True)
    parser.add_argument("--teacher_caption_embeddings", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_resize_size", type=int, default=256)
    parser.add_argument("--eval_crop_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--num_audit_updates",
        type=int,
        default=8,
        help="Number of fixed full GradCache updates for prior/plan diagnostics.",
    )
    parser.add_argument(
        "--loader_batch_size",
        type=int,
        default=16,
        help="DataLoader batch size in source images during SIGROT training.",
    )
    parser.add_argument(
        "--gradient_accum_steps",
        type=int,
        default=8,
        help="Number of loader batches concatenated into one GradCache transport plan.",
    )
    parser.add_argument(
        "--subsample_size",
        type=int,
        default=2000,
        help="Number of PAIRS sampled for student-vs-teacher graph recovery stats.",
    )
    parser.add_argument(
        "--max_train_gallery_queries",
        type=int,
        default=5000,
        help="Subsample size for text queries in the train-gallery retrieval check "
        "(full gallery is still all unique train images).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def build_eval_transforms(resize_size: int, crop_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(size=resize_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size=crop_size),
            v2.ToTensor(),
            v2.Normalize(mean=C.IMAGENET_DEFAULT_MEAN, std=C.IMAGENET_DEFAULT_STD),
        ]
    )


def determine_model_fmt(model_config: ViCLIPOTConfig) -> str:
    model_name = model_config.text_config.model_name
    if "gemma" in model_name.lower():
        return "gemma"
    elif "e5" in model_name.lower():
        return "e5"
    elif "qwen" in model_name.lower():
        return "qwen3"
    elif "bge" in model_name.lower():
        return "bge"
    elif "sbert" in model_name.lower():
        return "sbert"
    raise ValueError(f"Unsupported model name for determining model format: {model_name}")


class CollectedFeatures(NamedTuple):
    image_features: Tensor
    text_features: Tensor
    image_ids: Tensor
    pair_ids: Tensor
    batch_pair_counts: list[int]


@torch.inference_mode()
def collect_features(
    model: ViCLIPOT,
    data_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    device: torch.device,
) -> CollectedFeatures:
    """
    Preserve the exact pair IDs used by training for every emitted feature row.

    This matters for old UIT-OpenViIC copies: a truncated image can cause
    ImageTextDataset.__getitem__ to substitute the next readable sample. Static
    metadata positions then no longer match feature rows. New dataset code exposes
    batch["pair_ids"]; older code returns replacement indices, which map back through
    data_loader.dataset.get_pair_ids() to the same teacher rows.
    """
    all_image_features = []
    all_text_features = []
    all_image_ids = []
    all_pair_ids = []
    batch_pair_counts = []
    for batch in data_loader:
        if "pair_ids" in batch:
            pair_ids = torch.tensor(batch["pair_ids"], dtype=torch.int64)
        else:
            sample_indices = list(batch["indices"])
            pair_ids = torch.tensor(
                data_loader.dataset.get_pair_ids(sample_indices),
                dtype=torch.int64,  # pyright: ignore[reportAttributeAccessIssue]
            )
        images = batch["images"].to(device=device, non_blocking=True)
        text_inputs = batch["text_inputs"].to(device=device, non_blocking=True)
        model_outputs = model(images, text_inputs)
        image_features = model_outputs["image_features"].float().cpu()
        text_features = model_outputs["text_features"].float().cpu()
        image_ids = batch["image_ids"].cpu()
        num_rows = image_features.shape[0]
        if text_features.shape[0] != num_rows or image_ids.shape[0] != num_rows:
            raise RuntimeError("Collected image, text, and image-id row counts disagree.")
        if pair_ids.shape[0] != num_rows:
            raise RuntimeError(
                f"The collate emitted {pair_ids.shape[0]} pair IDs for {num_rows} feature rows."
            )
        all_image_features.append(image_features)
        all_text_features.append(text_features)
        all_image_ids.append(image_ids)
        all_pair_ids.append(pair_ids)
        batch_pair_counts.append(num_rows)
    return CollectedFeatures(
        image_features=torch.cat(all_image_features, dim=0),
        text_features=torch.cat(all_text_features, dim=0),
        image_ids=torch.cat(all_image_ids, dim=0),
        pair_ids=torch.cat(all_pair_ids, dim=0),
        batch_pair_counts=batch_pair_counts,
    )


def exact_chance_levels(image_ids: Tensor) -> dict[str, float]:
    """
    Analytic random-ranking reference levels matching `get_retrieval_metrics`.

    t2i protocol: every caption has exactly one correct image among N_unique ->
    P(R@k) = k / N_unique and expected rank (N_unique + 1) / 2.

    i2t protocol: a query image with c captions among N_total hits with hypergeometric
    probability 1 - C(N-c, k) / C(N, k); expected best-positive rank (N + 1) / (c + 1),
    averaged over unique-image queries like the evaluation does.
    Median-rank chance levels are intentionally omitted (finite-set discretization).
    """
    unique_ids, counts = np.unique(image_ids.numpy(), return_counts=True)
    num_unique_images = len(unique_ids)
    num_total_captions = int(counts.sum())

    def hypergeom_hit(captions_for_image: int, gallery: int, k: int) -> float:
        numerator = math.comb(gallery - captions_for_image, k)
        denominator = math.comb(gallery, k)
        if denominator == 0:
            # k exceeds gallery size; every ranking hits by construction
            return 1.0
        return 1.0 - numerator / denominator

    chance: dict[str, float] = {}
    for k in (1, 5, 10):
        chance[f"chance_t2i_R__{k}"] = min(k, num_unique_images) / num_unique_images
        i2t_k = min(k, num_total_captions)
        per_image_hits = [hypergeom_hit(int(c), num_total_captions, i2t_k) for c in counts]
        chance[f"chance_i2t_R__{k}"] = float(np.mean(per_image_hits))

    chance["chance_t2i_mean_rank"] = (num_unique_images + 1) / 2.0
    per_image_mean_ranks = [(num_total_captions + 1) / (int(c) + 1) for c in counts]
    chance["chance_i2t_mean_rank"] = float(np.mean(per_image_mean_ranks))
    chance["num_unique_test_images"] = num_unique_images
    chance["num_total_test_captions"] = num_total_captions
    return chance


def chunked_train_gallery_retrieval(
    text_features: Tensor,
    unique_image_features: Tensor,
    image_ids: Tensor,
    max_queries: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Text-to-image retrieval where the gallery is ALL unique train images.
    Measures memorization (high) vs transfer (low). Chunked to bound memory.
    """
    generator = torch.Generator().manual_seed(1234)
    num_texts = text_features.shape[0]
    if num_texts > max_queries:
        query_indices = torch.randperm(num_texts, generator=generator)[:max_queries]
    else:
        query_indices = torch.arange(num_texts)

    unique_ids, first_positions = np.unique(image_ids.numpy(), return_index=True)
    unique_ids_device = torch.from_numpy(unique_ids).to(device)
    unique_image_features = unique_image_features[torch.from_numpy(first_positions)].to(device)
    query_features = text_features[query_indices].to(device)
    query_image_ids = image_ids[query_indices].to(device)

    ranks = []
    chunk = 256
    for start in range(0, query_features.shape[0], chunk):
        sims = query_features[start : start + chunk] @ unique_image_features.t()
        positive_position = (
            query_image_ids[start : start + chunk].view(-1, 1) == unique_ids_device.view(1, -1)
        ).float()
        sorted_ranks = sims.argsort(dim=1, descending=True).argsort(dim=1).float()
        first_positive_rank = (sorted_ranks * positive_position).sum(dim=1) + 1.0
        ranks.append(first_positive_rank.cpu())
    ranks = torch.cat(ranks)

    gallery_size = len(unique_ids)
    return {
        "train_gallery_t2i_R__1": (ranks <= 1).float().mean().item(),
        "train_gallery_t2i_R__5": (ranks <= 5).float().mean().item(),
        "train_gallery_t2i_R__10": (ranks <= 10).float().mean().item(),
        "train_gallery_median_rank": math.floor(float(np.median(ranks.numpy()))),
        "train_gallery_mean_rank": float(np.mean(ranks.numpy())),
        "train_gallery_size": gallery_size,
        "train_gallery_num_queries": int(ranks.shape[0]),
        "train_gallery_chance_R__1": 1.0 / gallery_size,
    }


def combine_cross_modality(
    sim_matrix_text: Tensor,
    sim_matrix_image: Tensor,
    sim_matrix_text2image: Tensor,
    sim_matrix_image2text: Tensor,
) -> Tensor:
    return 0.25 * (
        sim_matrix_text + sim_matrix_image + sim_matrix_text2image + sim_matrix_image2text
    )


def audit_prior_and_plan(
    ot_loss: losses.BatchLevelEntropicOTLoss,
    logit_scale_value: float,
    teacher_image: Tensor,
    teacher_caption: Tensor,
    pair_ids: list[int],
    student_image: Tensor,
    student_text: Tensor,
) -> dict[str, float]:
    """
    Replicates the training-time computation for ONE micro-batch. All three inputs are
    pair-level: len(pair_ids) == student_image.shape[0] == student_text.shape[0].
    """
    assert len(pair_ids) == student_image.shape[0] == student_text.shape[0]

    sim_matrix_text = teacher_caption[pair_ids] @ teacher_caption[pair_ids].t()
    sim_matrix_image = teacher_image[pair_ids] @ teacher_image[pair_ids].t()
    sim_matrix_text2image = teacher_caption[pair_ids] @ teacher_image[pair_ids].t()
    sim_matrix_image2text = teacher_image[pair_ids] @ teacher_caption[pair_ids].t()
    combined_sim = combine_cross_modality(
        sim_matrix_text, sim_matrix_image, sim_matrix_text2image, sim_matrix_image2text
    )

    graph_prior = (logit_scale_value * combined_sim).softmax(dim=1)
    batch_size = graph_prior.shape[0]
    eye_mask = torch.eye(batch_size, dtype=torch.bool)
    diag_mass = graph_prior.diag().mean().item()
    row_entropy = -(graph_prior * (graph_prior + 1e-12).log()).sum(dim=1)
    off_diag_values = combined_sim.masked_select(~eye_mask)

    with torch.no_grad():
        raw_cosine_sim = student_image @ student_text.t()
        cost_matrix = 1 - raw_cosine_sim
        transport_plan = ot_loss.sinkhorn_unbalanced(
            cost_matrix, reg=0.05, reg_m=0.5, max_num_iters=200
        )
        row_sums = transport_plan.sum(dim=1)
        col_sums = transport_plan.sum(dim=0)
        raw_gkl_i2t = losses.generalized_kl_div(
            input=transport_plan, target=graph_prior, reduction="batchmean"
        ).item()
        raw_gkl_t2i = losses.generalized_kl_div(
            input=transport_plan.t(), target=graph_prior, reduction="batchmean"
        ).item()

    # Training reports loss_dict entries pre-halved (losses.py multiplies by 0.5) and
    # averaged per pair row; with reduction="sum"/num_rows that equals 0.5 * batchmean.
    return {
        "prior_diag_mass": diag_mass,
        "prior_row_entropy_nats": row_entropy.mean().item(),
        "prior_effective_support": row_entropy.exp().mean().item(),
        "combined_similarity_offdiag_mean": off_diag_values.mean().item(),
        "plan_total_mass": transport_plan.sum().item(),
        "plan_diag_mass_fraction": transport_plan.diag().sum().item()
        / transport_plan.sum().item(),
        "plan_row_sum_min": row_sums.min().item(),
        "plan_row_sum_max": row_sums.max().item(),
        "plan_marginal_error_mean_abs": (row_sums - 1.0).abs().mean().item(),
        "plan_col_marginal_error_mean_abs": (col_sums - 1.0).abs().mean().item(),
        "raw_gkl_i2t_batchmean": raw_gkl_i2t,
        "raw_gkl_t2i_batchmean": raw_gkl_t2i,
        "training_loss_i2t_component": 0.5 * raw_gkl_i2t,
        "training_loss_t2i_component": 0.5 * raw_gkl_t2i,
    }


def graph_recovery_stats(
    teacher_combined_sub: Tensor,
    student_combined_sub: Tensor,
    k: int = 10,
) -> dict[str, float]:
    batch_size = teacher_combined_sub.shape[0]
    eye_mask = torch.eye(batch_size, dtype=torch.bool)

    teacher_off = teacher_combined_sub.masked_select(~eye_mask)
    student_off = student_combined_sub.masked_select(~eye_mask)
    teacher_centered = teacher_off - teacher_off.mean()
    student_centered = student_off - student_off.mean()
    pearson = (
        (teacher_centered * student_centered).sum()
        / (teacher_centered.norm() * student_centered.norm() + 1e-12)
    ).item()

    masked_student = student_combined_sub.clone()
    masked_teacher = teacher_combined_sub.clone()
    masked_student.fill_diagonal_(-torch.inf)
    masked_teacher.fill_diagonal_(-torch.inf)
    top_student = masked_student.topk(k=k, dim=1).indices
    top_teacher = masked_teacher.topk(k=k, dim=1).indices
    overlaps = [
        len(set(top_student[row].tolist()) & set(top_teacher[row].tolist())) / k
        for row in range(batch_size)
    ]

    return {
        "offdiag_pearson_student_vs_teacher": pearson,
        f"top{k}_neighbor_overlap_student_vs_teacher": float(np.mean(overlaps)),
    }


def effective_rank(features: Tensor, max_samples: int = 2000) -> float:
    if features.shape[0] > max_samples:
        features = features[:max_samples]
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values**2
    return ((energy.sum() ** 2 / (energy**2).sum()) + 1e-12).item()


def validate_teacher_files(
    teacher_image: Tensor,
    teacher_caption: Tensor,
    metadata_pair_count: int,
    emitted_pair_ids: Tensor,
) -> dict[str, object]:
    """
    Validate files against metadata, then report what the loader actually emitted.

    Teacher tensors are made from metadata order, so their row count must match the
    metadata pair count. Emitted rows can be fewer when ImageTextDataset skips an
    unreadable image; that is valid because training indexes teachers with these exact
    emitted pair IDs.
    """
    if (
        teacher_caption.shape[0] != metadata_pair_count
        or teacher_image.shape[0] != metadata_pair_count
    ):
        raise ValueError(
            f"Teacher row count must equal the metadata pair count ({metadata_pair_count}), got "
            f"image={teacher_image.shape[0]}, caption={teacher_caption.shape[0]}. The teacher "
            "files were generated for a different dataset revision."
        )
    if emitted_pair_ids.numel() == 0:
        raise ValueError("The train loader emitted no pair IDs.")
    min_pair_id = emitted_pair_ids.min().item()
    max_pair_id = emitted_pair_ids.max().item()
    if min_pair_id < 0 or max_pair_id >= metadata_pair_count:
        raise ValueError(
            f"Emitted pair IDs must be within [0, {metadata_pair_count - 1}], got "
            f"min={min_pair_id}, max={max_pair_id}."
        )

    caption_norm_error = (teacher_caption.norm(dim=-1) - 1.0).abs().max().item()
    image_norm_error = (teacher_image.norm(dim=-1) - 1.0).abs().max().item()
    if caption_norm_error > 0.05 or image_norm_error > 0.05:
        raise ValueError(
            f"Teacher rows are not unit-norm (max deviation caption={caption_norm_error:.4f}, "
            f"image={image_norm_error:.4f}). Corrupted files should still be renormalized."
        )
    if not (torch.isfinite(teacher_image).all() and torch.isfinite(teacher_caption).all()):
        raise ValueError("Teacher embeddings contain non-finite values.")

    pair_id_counts = torch.bincount(emitted_pair_ids, minlength=metadata_pair_count)
    missing_pair_ids = (pair_id_counts == 0).nonzero().flatten()
    return {
        "teacher_image_shape": list(teacher_image.shape),
        "teacher_caption_shape": list(teacher_caption.shape),
        "metadata_pair_count": metadata_pair_count,
        "emitted_feature_rows": emitted_pair_ids.numel(),
        "unique_emitted_pair_ids": (pair_id_counts > 0).sum().item(),
        "duplicate_emitted_rows": (pair_id_counts - 1).clamp_min(0).sum().item(),
        "missing_metadata_pair_ids_count": missing_pair_ids.numel(),
        "missing_metadata_pair_ids_first_20": missing_pair_ids[:20].tolist(),
        "max_unit_norm_deviation_image": image_norm_error,
        "max_unit_norm_deviation_caption": caption_norm_error,
    }


def to_jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() > 1 else obj.item()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    args = parse_args()
    init_logger(level="INFO", compact=True)
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    logger.info(f"Auditing checkpoint {args.checkpoint} (arm={args.arm_label}) on device {device}")

    if args.num_audit_updates < 1:
        raise ValueError("--num_audit_updates must be at least 1.")
    if args.loader_batch_size < 1:
        raise ValueError("--loader_batch_size must be at least 1.")
    if args.gradient_accum_steps < 1:
        raise ValueError("--gradient_accum_steps must be at least 1.")

    model_config = ViCLIPOTConfig.model_validate(utils.load_yaml_file(args.model_config))
    model = ViCLIPOT(config=model_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_meta = {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "val_results": checkpoint.get("val_results"),
    }
    del checkpoint
    model.to(device).eval()

    logit_scale_value = model.logit_scale.exp().item()
    logit_bias_value = None if model.logit_bias is None else model.logit_bias.item()
    logger.info(
        f"Checkpoint epoch={checkpoint_meta['epoch']} global_step={checkpoint_meta['global_step']}"
    )
    logger.info(
        f"Actual learned logit_scale.exp() = {logit_scale_value:.4f} (logit_bias={logit_bias_value})"
    )
    checkpoint_meta.update(
        {
            "learned_logit_scale_exp": logit_scale_value,
            "logit_bias": logit_bias_value,
            "logit_scale_log_clamp_bounds": [math.log(1 / 100), math.log(100)],
        }
    )

    eval_transforms = build_eval_transforms(args.eval_resize_size, args.eval_crop_size)
    model_fmt = determine_model_fmt(model_config)
    tokenizer = model.text_encoder.tokenizer
    collate_fn = ImageTextCollate(
        tokenizer=tokenizer, max_length=model_config.max_length, caption_to_use="all"
    )

    test_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file="test.json",
        image_transforms=eval_transforms,
        model_fmt=model_fmt,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    logger.info("Encoding test split")
    test_features = collect_features(model, test_loader, device)
    test_retrieval_metrics = get_retrieval_metrics(
        image_features=test_features.image_features,
        text_features=test_features.text_features,
        logit_scale=torch.tensor(logit_scale_value),
        image_ids=test_features.image_ids,
    )
    chance_levels = exact_chance_levels(test_features.image_ids)

    signal_to_chance_t2i = test_retrieval_metrics["t2i_R__1"] / chance_levels["chance_t2i_R__1"]
    signal_to_chance_i2t = test_retrieval_metrics["i2t_R__1"] / chance_levels["chance_i2t_R__1"]
    logger.info(
        f"Test metrics: t2i_R@1={test_retrieval_metrics['t2i_R__1']:.6f} "
        f"i2t_R@1={test_retrieval_metrics['i2t_R__1']:.6f}"
    )
    logger.info(
        f"Chance levels: t2i={chance_levels['chance_t2i_R__1']:.6f} "
        f"i2t={chance_levels['chance_i2t_R__1']:.6f}"
    )
    logger.info(
        f"Signal-to-chance ratio: t2i x{signal_to_chance_t2i:.2f} i2t x{signal_to_chance_i2t:.2f}"
    )

    train_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file="train.json",
        image_transforms=eval_transforms,
        model_fmt=model_fmt,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.loader_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    logger.info("Encoding train split with deterministic transforms (this is the slow part)")
    train_features = collect_features(model, train_loader, device)

    train_gallery_results = chunked_train_gallery_retrieval(
        text_features=train_features.text_features,
        unique_image_features=train_features.image_features,
        image_ids=train_features.image_ids,
        max_queries=args.max_train_gallery_queries,
        device=device,
    )
    logger.info(
        "Train-gallery retrieval: "
        f"{json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in train_gallery_results.items()})}"
    )

    teacher_image = torch.load(args.teacher_image_embeddings, map_location="cpu").float()
    teacher_caption = torch.load(args.teacher_caption_embeddings, map_location="cpu").float()
    metadata_pair_count = sum(len(pair_ids) for pair_ids in train_dataset.pair_ids_by_sample_index)
    teacher_validation = validate_teacher_files(
        teacher_image, teacher_caption, metadata_pair_count, train_features.pair_ids
    )
    logger.info(
        f"Teacher embeddings validated: image {teacher_validation['teacher_image_shape']}, "
        f"caption {teacher_validation['teacher_caption_shape']}"
    )

    ot_loss = losses.BatchLevelEntropicOTLoss(sinkhorn_solver="sinkhorn_unbalanced")

    # GradCache concatenates eight loader batches before building one plan. Reuse the
    # emitted pair rows from those batches. This matches training even when the dataset
    # substitutes a readable sample for a truncated image.
    batch_row_offsets = np.concatenate([[0], np.cumsum(train_features.batch_pair_counts)])
    audit_batches: list[dict[str, float]] = []
    for update_number in range(args.num_audit_updates):
        start_batch = update_number * args.gradient_accum_steps
        end_batch = min(
            start_batch + args.gradient_accum_steps, len(train_features.batch_pair_counts)
        )
        if start_batch >= end_batch:
            break
        start_row = int(batch_row_offsets[start_batch])
        end_row = int(batch_row_offsets[end_batch])
        rows = torch.arange(start_row, end_row)
        pair_ids = train_features.pair_ids[rows].tolist()
        batch_stats = audit_prior_and_plan(
            ot_loss=ot_loss,
            logit_scale_value=logit_scale_value,
            teacher_image=teacher_image,
            teacher_caption=teacher_caption,
            pair_ids=pair_ids,
            student_image=train_features.image_features[rows],
            student_text=train_features.text_features[rows],
        )
        audit_batches.append(batch_stats)

    def mean_over_batches(key: str) -> float:
        return sum(stats[key] for stats in audit_batches) / len(audit_batches)

    plan_summary = {f"mean_{key}": mean_over_batches(key) for key in audit_batches[0]}
    logger.info(
        f"Prior/plan summary over {len(audit_batches)} batches: "
        f"{json.dumps({k: round(v, 6) for k, v in plan_summary.items()})}"
    )

    generator = torch.Generator().manual_seed(args.seed)
    num_subsample = min(args.subsample_size, train_features.pair_ids.numel())
    if num_subsample <= 10:
        raise ValueError("--subsample_size must select at least 11 pairs for top-10 overlap.")
    selected_rows = torch.randperm(train_features.pair_ids.numel(), generator=generator)[
        :num_subsample
    ]
    selected_pairs = train_features.pair_ids[selected_rows]
    teacher_combined_sub = combine_cross_modality(
        teacher_caption[selected_pairs] @ teacher_caption[selected_pairs].t(),
        teacher_image[selected_pairs] @ teacher_image[selected_pairs].t(),
        teacher_caption[selected_pairs] @ teacher_image[selected_pairs].t(),
        teacher_image[selected_pairs] @ teacher_caption[selected_pairs].t(),
    )
    student_combined_sub = combine_cross_modality(
        train_features.text_features[selected_rows]
        @ train_features.text_features[selected_rows].t(),
        train_features.image_features[selected_rows]
        @ train_features.image_features[selected_rows].t(),
        train_features.text_features[selected_rows]
        @ train_features.image_features[selected_rows].t(),
        train_features.image_features[selected_rows]
        @ train_features.text_features[selected_rows].t(),
    )
    recovery = graph_recovery_stats(teacher_combined_sub, student_combined_sub)
    selected_image_ids = train_features.image_ids[selected_rows]
    _unique_ids, first_positions = np.unique(selected_image_ids.numpy(), return_index=True)
    unique_image_rows = selected_rows[torch.from_numpy(first_positions)]
    recovery["student_image_effective_rank"] = effective_rank(
        train_features.image_features[unique_image_rows]
    )
    recovery["student_text_effective_rank"] = effective_rank(
        train_features.text_features[selected_rows]
    )
    logger.info(f"Graph recovery: {json.dumps({k: round(v, 6) for k, v in recovery.items()})}")

    results = {
        "arm_label": args.arm_label,
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "teacher_image_embeddings_path": os.path.abspath(args.teacher_image_embeddings),
        "teacher_caption_embeddings_path": os.path.abspath(args.teacher_caption_embeddings),
        "checkpoint_meta": checkpoint_meta,
        "computation_notes": {
            "feature_dtype": "float32",
            "transport_plan_device": "cpu",
            "note": "Training used bf16 autocast, train mode, and random crop/rotation on CUDA. "
            "This audit uses eval mode, deterministic transforms, and float32 Sinkhorn on CPU. "
            "It matches pair indexing and full-update plan size, but not per-step numerics exactly.",
        },
        "teacher_validation": teacher_validation,
        "test_metrics_recomputed": test_retrieval_metrics,
        "chance_levels_exact": chance_levels,
        "signal_to_chance_ratio": {
            "t2i_R__1": signal_to_chance_t2i,
            "i2t_R__1": signal_to_chance_i2t,
        },
        "train_gallery_retrieval": train_gallery_results,
        "prior_plan_audit": {
            "num_audit_updates": len(audit_batches),
            "loader_batch_size_samples": args.loader_batch_size,
            "gradient_accum_steps": args.gradient_accum_steps,
            "per_update": audit_batches,
            "summary": plan_summary,
        },
        "graph_recovery": recovery,
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w") as json_file:
        json.dump(to_jsonable(results), json_file, indent=2)
    logger.info(f"Wrote audit results to {args.output_json}")


if __name__ == "__main__":
    main()
