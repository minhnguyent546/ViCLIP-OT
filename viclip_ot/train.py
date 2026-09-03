# pyright: reportPrivateImportUsage=false

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.version
import torchvision.transforms.v2 as v2
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import viclip_ot.losses as losses
import viclip_ot.utils as utils
from viclip_ot.dataset import ImageTextCollate, ImageTextDataset
from viclip_ot.model_factory import create_training_model
from viclip_ot.opts import add_training_opts
from viclip_ot.utils.logger import init_logger, logger
from viclip_ot.utils.metric import AverageMeter
from viclip_ot.utils.training import (
    EarlyStopping,
    EvalResults,
    RandContext,
    eval_model,
    get_parameter_names,
    maybe_log_eval_results,
    print_eval_results,
    save_top_k_checkpoints,
)


def train_model(args: argparse.Namespace) -> None:
    checkpoint_dir = None
    log_file = None
    if not args.run_test_only:
        checkpoints_dir_basename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.wandb_logging and args.wandb_name is not None:
            checkpoints_dir_basename += f"-{args.wandb_name}"

        checkpoint_dir = os.path.join(
            args.checkpoints_dir,
            checkpoints_dir_basename,
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        if args.wandb_logging and args.wandb_name is not None:
            log_file = os.path.join(checkpoint_dir, f"{args.wandb_name}.log")
        else:
            log_file = os.path.join(checkpoint_dir, "train.log")

    # init logger
    logger_init_config = init_logger(level="DEBUG", log_file=log_file, compact=True)

    # set random seed
    utils.set_seed(args.seed)

    # log system information and args
    git_info = utils.get_git_info()
    if git_info.get("repository") is not None:
        logger.info(f"Git repository: {git_info['repository']}")
    if git_info.get("commit_hash") is not None:
        logger.info(f"Git commit hash: {git_info['commit_hash'][:7]}")
    if git_info.get("branch") is not None:
        logger.info(f"Git branch: {git_info['branch']}")
    logger.info(f"Python version: {sys.version}")
    logger.info(
        f"Pytorch version {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
    )
    logger.info(
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout
    )
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Args: {args}")
    logger.info(f"Effective batch size: {args.train_batch_size * args.gradient_accum_steps}")

    # training device
    device = utils.get_device(args.device)
    logger.info(f"Using device: {device}")

    # creating model
    model_bundle = create_training_model(args.model_config)
    model_config = model_bundle.config
    logger.info(f"Model config: {model_config}")
    model = model_bundle.model
    tokenizer = model_bundle.tokenizer
    logger.info(f"Model class: {type(model).__name__}")
    model.to(device)

    # loss fun
    caption_embeddings = None
    image_embeddings = None
    sim_graph_alpha = args.sim_graph_alpha
    sim_combine_method = args.sim_combine_method
    if args.sim_graph_regularized_ot and args.criterion in (
        "batch_level_entropic_ot_loss",
        "direct_graph_kl_loss",
        "hybrid_clip_tp_loss",
        "hybrid_sig_lip_tp_loss",
        "hybrid_clip_direct_graph_kl_loss",
        "hybrid_sig_lip_direct_graph_kl_loss",
    ):
        if (
            args.precomputed_caption_embeddings_path is None
            or not os.path.isfile(args.precomputed_caption_embeddings_path)
            or not args.precomputed_caption_embeddings_path.endswith(".pt")
        ):
            raise ValueError(
                "Please provide a valid path to precomputed caption embeddings (.pt file) "
                "when using similarity graph regularized OT."
            )
        logger.info(
            f"Using similarity graph regularized OT with precomputed caption embeddings from {args.precomputed_caption_embeddings_path}.",
        )
        caption_embeddings = torch.load(
            args.precomputed_caption_embeddings_path, map_location="cpu"
        )  # already normalized

        if (
            args.precomputed_image_embeddings_path is None
            or not os.path.isfile(args.precomputed_image_embeddings_path)
            or not args.precomputed_image_embeddings_path.endswith(".pt")
        ):
            raise ValueError(
                "Please provide a valid path to precomputed image embeddings (.pt file) "
                "when using similarity graph regularized OT."
            )
        logger.info(
            f"Using similarity graph regularized OT with precomputed image embeddings from {args.precomputed_image_embeddings_path}.",
        )
        image_embeddings = torch.load(
            args.precomputed_image_embeddings_path, map_location="cpu"
        )  # already normalized

        logger.info(
            f"Loaded caption embeddings with shape {caption_embeddings.shape} "
            f"and image embeddings with shape {image_embeddings.shape}."
        )

        logger.info(
            f"Using precomputed similarity graph with alpha (image embeddings weight) = {args.sim_graph_alpha}."
        )

        logger.info(f"#### Combining similarity matrices using method: {args.sim_combine_method}")

    def _combine_sim_matrices(
        sim_matrix_text: torch.Tensor,
        sim_matrix_image: torch.Tensor,
        sim_matrix_text2image: torch.Tensor | None = None,
        sim_matrix_image2text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Method to combine similarity graphs from image and text modalities: ["weighted_sum", "maximum", "harmonic_mean", "sparse_thresholding", "minimum", "power_mean", "arithmetic_mean", "cross_modality"]
        """
        if sim_combine_method == "weighted_sum":
            if sim_graph_alpha is None:
                raise ValueError("sim_graph_alpha must be provided for weighted_sum method.")
            return (1 - sim_graph_alpha) * sim_matrix_text + sim_graph_alpha * sim_matrix_image
        elif sim_combine_method == "maximum":
            return torch.maximum(sim_matrix_text, sim_matrix_image)
        elif sim_combine_method == "harmonic_mean":
            return (
                2
                * (sim_matrix_text * sim_matrix_image)
                / (sim_matrix_text + sim_matrix_image + 1e-8)
            )
        elif sim_combine_method == "sparse_thresholding":
            if sim_graph_alpha is None:
                raise ValueError(
                    "sim_graph_alpha must be provided for sparse_thresholding method."
                )
            combined_sim = (
                1 - sim_graph_alpha
            ) * sim_matrix_text + sim_graph_alpha * sim_matrix_image
            threshold = torch.quantile(combined_sim, args.sim_sparse_threshold_quantile)
            combined_sim[combined_sim < threshold] = 0.0
            return combined_sim
        elif sim_combine_method == "minimum":
            return torch.minimum(sim_matrix_text, sim_matrix_image)
        elif sim_combine_method == "power_mean":
            p = args.sim_power_mean_exponent  # e.g., p=3
            return ((sim_matrix_text**p + sim_matrix_image**p) / 2) ** (1 / p)
        elif sim_combine_method == "arithmetic_mean":
            return (sim_matrix_text + sim_matrix_image) / 2
        elif sim_combine_method == "cross_modality":
            # 1/4 * (text-text + image-image + text-image + image-text)
            assert sim_matrix_text2image is not None and sim_matrix_image2text is not None, (
                "sim_matrix_text2image and sim_matrix_image2text must be provided for cross_modality method."
            )
            return 0.25 * (
                sim_matrix_text + sim_matrix_image + sim_matrix_text2image + sim_matrix_image2text
            )
        elif sim_combine_method == "cross_modality3":
            # 1/3 * (text-text + text-image + image-text)
            assert sim_matrix_text2image is not None and sim_matrix_image2text is not None, (
                "sim_matrix_text2image and sim_matrix_image2text must be provided for cross_modality method."
            )
            return (1 / 3) * (sim_matrix_text + sim_matrix_text2image + sim_matrix_image2text)
        else:
            raise ValueError(f"Unsupported sim_combine_method: {sim_combine_method}")

    if args.criterion == "clip_loss":
        criterion = losses.ClipLoss()
    elif args.criterion == "sig_lip_loss":
        criterion = losses.SigLipLoss()
    elif args.criterion == "batch_level_entropic_ot_loss":
        criterion = losses.BatchLevelEntropicOTLoss(
            sinkhorn_solver=args.sinkhorn_solver,
            use_transport_plan_as_logits=args.use_transport_plan_as_logits,
            sim_matrix_temperature=args.sim_matrix_temperature,
        )
    elif args.criterion == "direct_graph_kl_loss":
        criterion = losses.DirectGraphKLLoss()
    elif args.criterion == "hybrid_clip_tp_loss":
        criterion = losses.HybridClipTPLoss(
            clip_loss_lambda=args.hybrid_clip_tp_loss_clip_loss_lambda,
            sinkhorn_solver=args.sinkhorn_solver,
            use_transport_plan_as_logits=args.use_transport_plan_as_logits,
            sim_matrix_temperature=args.sim_matrix_temperature,
        )
    elif args.criterion == "hybrid_sig_lip_tp_loss":
        criterion = losses.HybridSigLipTPLoss(
            sig_lip_loss_lambda=args.hybrid_sig_lip_tp_loss_sig_lip_loss_lambda,
            sinkhorn_solver=args.sinkhorn_solver,
            use_transport_plan_as_logits=args.use_transport_plan_as_logits,
            sim_matrix_temperature=args.sim_matrix_temperature,
        )
    elif args.criterion == "hybrid_clip_direct_graph_kl_loss":
        criterion = losses.HybridClipDirectGraphKLLoss(
            clip_loss_lambda=args.hybrid_clip_direct_graph_kl_loss_clip_loss_lambda,
        )
    elif args.criterion == "hybrid_sig_lip_direct_graph_kl_loss":
        criterion = losses.HybridSigLipDirectGraphKLLoss(
            sig_lip_loss_lambda=args.hybrid_sig_lip_direct_graph_kl_loss_sig_lip_loss_lambda,
        )
    else:
        raise ValueError(f"Unsupported criterion: {args.criterion}")

    if args.eval_criterion == "clip_loss":
        eval_criterion = losses.ClipLoss()
    elif args.eval_criterion == "sig_lip_loss":
        eval_criterion = losses.SigLipLoss()
    else:
        raise ValueError(f"Unsupported evaluation criterion: {args.criterion}")

    if args.linear_probing:
        raise NotImplementedError("Loading from checkpoint is not implemented yet.")

    def load_model_checkpoint_state(
        state_dict: dict[str, Tensor],
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        if model_bundle.uses_checkpoint_metadata:
            model.validate_checkpoint_metadata(  # pyright: ignore[reportAttributeAccessIssue]
                checkpoint_metadata
            )
        if model_bundle.save_trainable_state_only:
            model.load_checkpoint_state_dict(  # pyright: ignore[reportAttributeAccessIssue]
                state_dict
            )
        else:
            model.load_state_dict(state_dict, strict=True)

    if args.from_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {args.from_checkpoint}")
        checkpoint = torch.load(args.from_checkpoint, map_location=device, weights_only=False)
        load_model_checkpoint_state(
            checkpoint["model_state_dict"],
            checkpoint.get("model_metadata"),
        )

    if not model_bundle.supports_tower_locking and (args.lock_image or args.lock_text):
        raise ValueError("The selected model does not support --lock_image or --lock_text.")
    if args.lock_image:
        model.lock_image_tower(  # pyright: ignore[reportAttributeAccessIssue]
            last_unfreeze_groups=args.lock_image_last_unfreeze_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats,
        )
    if args.lock_text:
        model.lock_text_tower(  # pyright: ignore[reportAttributeAccessIssue]
            unfreeze_dense=args.lock_text_unfreeze_dense
        )

    if model_bundle.required_image_size is not None:
        if args.train_crop_size != model_bundle.required_image_size:
            raise ValueError(
                f"The selected model requires --train_crop_size "
                f"{model_bundle.required_image_size}, found {args.train_crop_size}."
            )
        if args.eval_crop_size != model_bundle.required_image_size:
            raise ValueError(
                f"The selected model requires --eval_crop_size "
                f"{model_bundle.required_image_size}, found {args.eval_crop_size}."
            )
        if (
            model_bundle.eval_resize_mode == "direct_square"
            and args.eval_resize_size != model_bundle.required_image_size
        ):
            raise ValueError(
                f"The selected model requires --eval_resize_size "
                f"{model_bundle.required_image_size}, found {args.eval_resize_size}."
            )
        if (
            model_bundle.eval_resize_mode == "resize_then_crop"
            and args.eval_resize_size < model_bundle.required_image_size
        ):
            raise ValueError(
                f"--eval_resize_size must be at least {model_bundle.required_image_size} "
                "for the selected model."
            )

    # loading dataset
    train_transforms = v2.Compose(
        [
            v2.RandomRotation(15),  # pyright: ignore[reportArgumentType]
            v2.RandomResizedCrop(
                size=args.train_crop_size,
                scale=(0.8, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            v2.ToTensor(),
            v2.Normalize(
                mean=model_bundle.image_mean,
                std=model_bundle.image_std,
            ),
        ]
    )
    if model_bundle.eval_resize_mode == "direct_square":
        eval_resize = v2.Resize(
            size=(args.eval_resize_size, args.eval_resize_size),
            interpolation=v2.InterpolationMode.BICUBIC,
        )
        eval_geometry_transforms = [eval_resize]
    elif model_bundle.eval_resize_mode == "resize_then_crop":
        eval_geometry_transforms = [
            v2.Resize(size=args.eval_resize_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size=args.eval_crop_size),
        ]
    else:
        raise ValueError(
            f"Unsupported eval_resize_mode: {model_bundle.eval_resize_mode}. "
            "Supported modes are: 'direct_square', 'resize_then_crop'."
        )
    eval_transforms = v2.Compose(
        [
            *eval_geometry_transforms,
            v2.ToTensor(),
            v2.Normalize(
                mean=model_bundle.image_mean,
                std=model_bundle.image_std,
            ),
        ]
    )

    logger.info(f"Determined model format: {model_bundle.caption_format}")

    train_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file=f"{args.train_split_name}.json",
        image_transforms=train_transforms,
        model_fmt=model_bundle.caption_format,
    )
    test_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file=f"{args.test_split_name}.json",
        image_transforms=eval_transforms,
        model_fmt=model_bundle.caption_format,
    )
    val_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file=f"{args.val_split_name}.json",
        image_transforms=eval_transforms,
        model_fmt=model_bundle.caption_format,
    )

    logger.info(
        f"train_size = {len(train_dataset)}, "
        f"test_size = {len(test_dataset)}, "
        f"val_size = {len(val_dataset)} "
    )

    train_collate_fun = ImageTextCollate(
        tokenizer=tokenizer,
        max_length=model_bundle.max_length,
        padding=model_bundle.text_padding,
        caption_to_use=args.train_caption_to_use,
        pair_ids_by_sample_index=train_dataset.pair_ids_by_sample_index,
    )
    eval_collate_fun = ImageTextCollate(
        tokenizer=tokenizer,
        max_length=model_bundle.max_length,
        padding=model_bundle.text_padding,
        caption_to_use="all",
        pair_ids_by_sample_index=val_dataset.pair_ids_by_sample_index,
    )
    test_collate_fun = ImageTextCollate(
        tokenizer=tokenizer,
        max_length=model_bundle.max_length,
        padding=model_bundle.text_padding,
        caption_to_use="all",
        pair_ids_by_sample_index=test_dataset.pair_ids_by_sample_index,
    )
    # creating data loaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_collate_fun,
        persistent_workers=args.num_workers > 0,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fun,
        persistent_workers=args.num_workers > 0,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=eval_collate_fun,
        persistent_workers=args.num_workers > 0,
    )

    # mixed precision training
    mp_dtype = torch.float32
    if device.type == "cuda" and args.mixed_precision == "fp16":
        mp_dtype = torch.float16
    elif device.type == "cuda" and args.mixed_precision == "bf16":
        if torch.cuda.is_bf16_supported():
            mp_dtype = torch.bfloat16
        else:
            mp_dtype = torch.float16
    if mp_dtype != torch.float32:
        logger.info(f"Mixed precision training enabled with dtype {mp_dtype}")

    autocast_context = torch.autocast(
        device_type=device.type,
        dtype=mp_dtype,
        enabled=(mp_dtype in (torch.float16, torch.bfloat16)),
    )
    scaler = torch.amp.grad_scaler.GradScaler(
        device=device.type,
        enabled=(mp_dtype == torch.float16),
    )

    # setting up logging with wandb
    wandb_run = None
    if args.wandb_logging:
        if "WANDB_API_KEY" not in os.environ:
            raise RuntimeError(
                "WANDB_API_KEY environment variable is not set. "
                "Please set it to enable wandb logging."
            )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "training_args": vars(args),
                "model_config": model_config.model_dump(),
            },
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            id=args.wandb_resume_id,
            resume="must" if args.wandb_resume_id is not None else None,
        )
        wandb_run.define_metric(name="val/*", step_metric="epoch")
        wandb_run.define_metric(name="test/*", step_metric="epoch")
        wandb_run.define_metric(name="train/epoch_loss", step_metric="epoch")

    if not args.run_test_only:
        assert checkpoint_dir is not None
        utils.save_metadata_to_checkpoint(
            checkpoint_dir=checkpoint_dir,
            args=args,
            model_config=model_config.model_dump(),
            wandb_run=wandb_run,
        )

    num_model_params = utils.count_model_params(model, trainable=False)
    num_model_trainable_params = utils.count_model_params(model, trainable=True)
    logger.info(
        f"num_params: {utils.to_human_readable(num_model_params)} | num_trainable_params: {utils.to_human_readable(num_model_trainable_params)}"
    )

    def _run_test_only(data_loader: DataLoader) -> None:  # pyright: ignore[reportMissingTypeArgument]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        test_start_time = time.perf_counter()
        test_criterion = (
            eval_criterion
            if model_bundle.test_only_uses_eval_criterion
            or isinstance(
                criterion,
                (
                    losses.DirectGraphKLLoss,
                    losses.HybridClipDirectGraphKLLoss,
                    losses.HybridSigLipDirectGraphKLLoss,
                ),
            )
            else criterion
        )
        logger.info(f"Test loss criterion: {type(test_criterion).__name__}")
        test_results = eval_model(
            model=model,
            criterion=test_criterion,
            eval_data_loader=data_loader,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        test_elapsed_time = time.perf_counter() - test_start_time
        logger.info(
            "** Test results **\n"
            f"    loss: {test_results['loss']:0.6f}\n"
            "     Text to image:\n"
            f"        t2i_R__1: {test_results['t2i_R__1']:0.6f}\n"
            f"        t2i_R__5: {test_results['t2i_R__5']:0.6f}\n"
            f"        t2i_R__10: {test_results['t2i_R__10']:0.6f}\n"
            f"        t2i_mean_rank: {test_results['t2i_mean_rank']:0.6f}\n"
            f"        t2i_median_rank: {test_results['t2i_median_rank']:0.6f}\n"
            "     Image to text:\n"
            f"      i2t_R__1: {test_results['i2t_R__1']:0.6f}\n"
            f"      i2t_R__5: {test_results['i2t_R__5']:0.6f}\n"
            f"      i2t_R__10: {test_results['i2t_R__10']:0.6f}\n"
            f"      i2t_mean_rank: {test_results['i2t_mean_rank']:0.6f}\n"
            f"      i2t_median_rank: {test_results['i2t_median_rank']:0.6f}\n"
            f"    Alignment_score: {test_results['alignment_score']:0.6f}\n"
            f"    Modality_gap: {test_results['modality_gap']:0.6f}\n"
            f"    Elapsed time: {utils.to_hms(test_elapsed_time)}\n"
        )

    if args.run_test_only:
        return _run_test_only(test_data_loader)

    assert checkpoint_dir is not None

    decay_parameters = set(
        get_parameter_names(
            model,
            forbidden_layer_types=[
                nn.LayerNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.GroupNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
                nn.Embedding,
            ],
            forbidden_layer_names=["bias", "norm", "logit_scale", "logit_bias"],
        )
    )

    backbone_prefixes = model_bundle.backbone_prefixes
    adapters_prefixes = model_bundle.adapter_prefixes

    def has_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
        return any(name.startswith(p) for p in prefixes)

    def collect(prefixes: tuple[str, ...], do_decay: bool):
        return [
            p
            for n, p in model.named_parameters()
            if p.requires_grad
            and has_prefix(n, prefixes)
            and ((n in decay_parameters) == do_decay)
        ]

    param_groups = [
        {
            "params": collect(backbone_prefixes, do_decay=True),
            "weight_decay": args.weight_decay,
            "lr": args.backbone_lr,
            "name": "decay__backbone",
        },
        {
            "params": collect(backbone_prefixes, do_decay=False),
            "weight_decay": 0.0,
            "lr": args.backbone_lr,
            "name": "no_decay__backbone",
        },
        {
            "params": collect(adapters_prefixes, do_decay=True),
            "weight_decay": args.weight_decay,
            "lr": args.lr,
            "name": "decay__adapters",
        },
        {
            "params": collect(adapters_prefixes, do_decay=False),
            "weight_decay": 0.0,
            "lr": args.lr,
            "name": "no_decay__adapters",
        },
    ]
    for i, param_group in enumerate(param_groups):
        logger.debug(
            f"param_group: {param_group.get('name', i)} | num_params: {len(param_group['params'])}"  # pyright: ignore[reportArgumentType]
        )
    # --- sanity checks (catch silent bugs) ---
    all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    grouped = {id(p) for g in param_groups for p in g["params"]}  # pyright: ignore[reportGeneralTypeIssues]

    missing = all_trainable - grouped
    extra = grouped - all_trainable

    if missing:
        logger.error("Missing trainable params:")
        for n, p in model.named_parameters():
            if p.requires_grad and id(p) in missing:
                logger.info("  ", n)

    if extra:
        logger.error("Frozen params in optimizer:")
        for n, p in model.named_parameters():
            if not p.requires_grad and id(p) in extra:
                logger.info("  ", n)

    assert not missing, f"Trainable params missing from optimizer: {len(missing)}"
    assert not extra, f"Frozen params included in optimizer: {len(extra)}"

    optim_cls = None
    if args.optimizer == "adam":
        optim_cls = torch.optim.Adam
    elif args.optimizer == "adamw":
        optim_cls = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    logger.info(
        f"Using {args.optimizer} optimizer with eps={args.adam_eps} and betas={args.adam_betas}"
    )
    optimizer = optim_cls(param_groups, betas=args.adam_betas, eps=args.adam_eps)

    num_updates_per_epoch = (
        len(train_data_loader) + args.gradient_accum_steps - 1
    ) // args.gradient_accum_steps
    if args.scheduler == "cosine_annealing":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=args.cosine_annealing_T_0,
            T_mult=args.cosine_annealing_T_mult,
            eta_min=args.min_lr,
        )
    elif args.scheduler == "one_cycle_lr":
        max_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        main_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lrs,
            pct_start=0.05,  # warmup is now can be used via a separate scheduler
            epochs=args.num_epochs - args.lr_warmup_epochs,
            steps_per_epoch=num_updates_per_epoch,
            div_factor=1.0,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    # learning rate warmup
    warmup_lr_scheduler = None
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs * num_updates_per_epoch,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs * num_updates_per_epoch,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only `linear` and `constant` are supported."
            )

    optimizer.zero_grad()
    if args.max_grad_norm > 0:
        logger.info(f"Using gradient clipping with max norm {args.max_grad_norm}")

    # results for each metric will be sorted in decreasing order
    # metric with prefix '_' indicates lower is better (e.g. _loss)
    eval_results_keys = list(EvalResults.__annotations__.keys())
    for metric in args.best_checkpoint_metrics:
        assert metric.lstrip("_") in eval_results_keys, (
            f"Metric {metric} is not a valid metric, possible metrics: {eval_results_keys}"
        )

    # best_val_results[metric] = list of tuples (value, checkpoint_path)
    best_val_results: dict[str, list[tuple[float, str]]] = {
        metric.lstrip("_"): [] for metric in args.best_checkpoint_metrics
    }
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, min_delta=0.0, enabled=args.early_stopping
    )
    if early_stopping.is_enabled():
        logger.info(f"Early stopping enabled with patience {early_stopping.patience}")

    # disable logging to stdout during training to avoid conflict with tqdm
    logger.remove(logger_init_config["stdout_id"])
    global_step = 0
    total_training_only_time = 0.0
    total_eval_time = 0.0
    total_training_pairs = 0
    for epoch in range(args.num_epochs):
        model.train()
        criterion_kwargs = {}

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_training_start_time = time.perf_counter()

        train_data_iter = iter(train_data_loader)
        total_num_samples = len(train_data_loader)
        last_iter_num_batches = total_num_samples % args.gradient_accum_steps
        if last_iter_num_batches == 0:
            last_iter_num_batches = args.gradient_accum_steps

        # determine the number of updates for the current epoch
        # based on gradient accumulation steps
        num_updates_per_epoch = (
            total_num_samples + args.gradient_accum_steps - 1
        ) // args.gradient_accum_steps  # ceil_div

        train_progressbar = tqdm(
            range(num_updates_per_epoch),
            desc=f"Training epoch {epoch + 1}/{args.num_epochs}",
        )

        train_loss = AverageMeter(name="train_loss", fmt=":0.4f")
        epoch_training_pairs = 0

        for update_step in train_progressbar:
            num_batches = (
                args.gradient_accum_steps
                if update_step + 1 < num_updates_per_epoch
                else last_iter_num_batches
            )
            batches, num_items_in_batch = utils.get_batch_samples(
                data_iter=train_data_iter,
                num_batches=num_batches,
                labels_key="images",
            )
            assert num_items_in_batch is not None
            num_batches = len(batches)  # actual number batches retrieved

            batch_loss: float = 0.0
            batch_loss_components: dict[str, float] = {}

            if num_batches == 1:
                images = batches[0]["images"].to(device=device, non_blocking=True)
                text_inputs = batches[0]["text_inputs"].to(device=device, non_blocking=True)
                image_ids = batches[0]["image_ids"]

                if (
                    isinstance(
                        criterion,
                        (
                            losses.BatchLevelEntropicOTLoss,
                            losses.DirectGraphKLLoss,
                            losses.HybridClipTPLoss,
                            losses.HybridSigLipTPLoss,
                            losses.HybridClipDirectGraphKLLoss,
                            losses.HybridSigLipDirectGraphKLLoss,
                        ),
                    )
                    and caption_embeddings is not None
                    and image_embeddings is not None
                ):
                    pair_ids = batches[0]["pair_ids"]
                    batch_caption_embeddings = caption_embeddings[pair_ids].to(
                        device=device, non_blocking=True
                    )
                    batch_image_embeddings = image_embeddings[pair_ids].to(
                        device=device, non_blocking=True
                    )

                    sim_matrix_text = batch_caption_embeddings @ batch_caption_embeddings.t()
                    sim_matrix_image = batch_image_embeddings @ batch_image_embeddings.t()
                    sim_matrix_text2image = batch_caption_embeddings @ batch_image_embeddings.t()
                    sim_matrix_image2text = batch_image_embeddings @ batch_caption_embeddings.t()

                    sim_matrix = _combine_sim_matrices(
                        sim_matrix_text,
                        sim_matrix_image,
                        sim_matrix_text2image,
                        sim_matrix_image2text,
                    )

                    criterion_kwargs["sim_matrix"] = sim_matrix

                with autocast_context:
                    model_outputs = model(images, text_inputs)
                    loss_dict = criterion(
                        image_features=model_outputs["image_features"],
                        text_features=model_outputs["text_features"],
                        logit_scale=model_outputs["logit_scale"],
                        logit_bias=model_outputs.get("logit_bias", None),
                        image_ids=image_ids,
                        reduction="sum",
                        output_dict=True,
                        **criterion_kwargs,
                    )
                    loss = sum(loss_dict.values())
                    if num_items_in_batch > 0:
                        loss = loss / num_items_in_batch

                scaler.scale(loss).backward()
                batch_loss += loss.detach().item()
                for name, value in loss_dict.items():
                    component_value = value.detach().item()
                    if num_items_in_batch > 0:
                        component_value /= num_items_in_batch
                    batch_loss_components[name] = (
                        batch_loss_components.get(name, 0.0) + component_value
                    )
            else:
                # GradCache (loss once on full batch; encoder grads via surrogate).
                moved_batches: list[dict] = []
                cached_image_features: list[Tensor] = []
                cached_text_features: list[Tensor] = []
                random_states: list[RandContext] = []

                # Step 1: no-grad encode each micro-batch; cache features + RNG
                # (snapshot only — do not wrap this forward in RandContext).
                with torch.no_grad():
                    for batch in batches:
                        images = batch["images"].to(device=device, non_blocking=True)
                        text_inputs = batch["text_inputs"].to(device=device, non_blocking=True)
                        rnd = RandContext(images, *text_inputs.values())
                        with autocast_context:
                            model_outputs = model(images, text_inputs)
                        moved_batches.append(
                            {
                                "images": images,
                                "text_inputs": text_inputs,
                                "image_ids": batch["image_ids"],
                                "pair_ids": batch["pair_ids"],
                            }
                        )
                        cached_image_features.append(model_outputs["image_features"])
                        cached_text_features.append(model_outputs["text_features"])
                        random_states.append(rnd)

                all_image_ids = torch.cat([batch["image_ids"] for batch in moved_batches], dim=0)

                if (
                    isinstance(
                        criterion,
                        (
                            losses.BatchLevelEntropicOTLoss,
                            losses.DirectGraphKLLoss,
                            losses.HybridClipTPLoss,
                            losses.HybridSigLipTPLoss,
                            losses.HybridClipDirectGraphKLLoss,
                            losses.HybridSigLipDirectGraphKLLoss,
                        ),
                    )
                    and caption_embeddings is not None
                    and image_embeddings is not None
                ):
                    pair_ids = [
                        pair_id for batch in moved_batches for pair_id in batch["pair_ids"]
                    ]
                    batch_caption_embeddings = caption_embeddings[pair_ids].to(
                        device=device, non_blocking=True
                    )
                    batch_image_embeddings = image_embeddings[pair_ids].to(
                        device=device, non_blocking=True
                    )

                    sim_matrix_text = batch_caption_embeddings @ batch_caption_embeddings.t()
                    sim_matrix_image = batch_image_embeddings @ batch_image_embeddings.t()
                    sim_matrix_text2image = batch_caption_embeddings @ batch_image_embeddings.t()
                    sim_matrix_image2text = batch_image_embeddings @ batch_caption_embeddings.t()

                    sim_matrix = _combine_sim_matrices(
                        sim_matrix_text,
                        sim_matrix_image,
                        sim_matrix_text2image,
                        sim_matrix_image2text,
                    )

                    criterion_kwargs["sim_matrix"] = sim_matrix

                # Step 2: one loss on detached full-batch features → cache ∂L/∂z
                # (also grads logit_scale / logit_bias; Sinkhorn runs once).
                image_features = torch.cat(cached_image_features, dim=0).detach().requires_grad_()
                text_features = torch.cat(cached_text_features, dim=0).detach().requires_grad_()
                logit_scale = model.logit_scale.exp()
                logit_bias = model.logit_bias if model.logit_bias is not None else None

                with autocast_context:
                    loss_dict = criterion(
                        image_features=image_features,
                        text_features=text_features,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        image_ids=all_image_ids,
                        reduction="sum",
                        output_dict=True,
                        **criterion_kwargs,
                    )
                    loss = sum(loss_dict.values())
                    if num_items_in_batch > 0:
                        loss = loss / num_items_in_batch

                scaler.scale(loss).backward()

                image_grad_chunks = list(
                    image_features.grad.split(
                        [batch["images"].shape[0] for batch in moved_batches]
                    )
                )
                text_grad_chunks = list(
                    text_features.grad.split([batch["images"].shape[0] for batch in moved_batches])
                )

                batch_loss = loss.detach().item()
                for name, value in loss_dict.items():
                    component_value = value.detach().item()
                    if num_items_in_batch > 0:
                        component_value /= num_items_in_batch
                    batch_loss_components[name] = component_value

                del image_features, text_features, loss_dict, loss
                del cached_image_features, cached_text_features

                # Step 3: second encode per micro-batch; surrogate ⟨z, ∂L/∂z⟩ pushes
                # cached feature grads into the encoder (grads already scaler-scaled).
                for batch, rnd, image_grad, text_grad in zip(
                    moved_batches,
                    random_states,
                    image_grad_chunks,
                    text_grad_chunks,
                    strict=True,
                ):
                    with rnd:
                        with autocast_context:
                            model_outputs = model(batch["images"], batch["text_inputs"])
                            surrogate = torch.dot(
                                model_outputs["image_features"].flatten(),
                                image_grad.flatten(),
                            ) + torch.dot(
                                model_outputs["text_features"].flatten(),
                                text_grad.flatten(),
                            )
                    surrogate.backward()

                del moved_batches, random_states, image_grad_chunks, text_grad_chunks

            grad_norm_value = 0.0
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm_value = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.max_grad_norm,
                    norm_type=2,
                )
                if not bool(torch.isinf(grad_norm_value)) and not bool(
                    torch.isnan(grad_norm_value)
                ):
                    grad_norm_value = grad_norm_value.item()
                else:
                    grad_norm_value = 0.0

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_training_pairs += num_items_in_batch

            # Clamp log(1 / temperature) after each optimizer update, as in CLIP.
            with torch.no_grad():
                model.logit_scale.clamp_(
                    min=np.log(model_bundle.logit_scale_min),
                    max=np.log(model_bundle.logit_scale_max),
                )

            # Skip a lone "loss" component (e.g. plain SigLIP) — it duplicates the total.
            detailed_loss_components = {
                name: value
                for name, value in batch_loss_components.items()
                if not (name == "loss" and len(batch_loss_components) == 1)
            }

            if wandb_run is not None:
                log_data = {
                    f"learning_rate/group_{param_group.get('name', group_id)}": param_group["lr"]
                    for group_id, param_group in enumerate(optimizer.param_groups)
                }
                log_data["train/loss"] = batch_loss
                for name, value in detailed_loss_components.items():
                    log_data[f"train/{name}"] = value
                log_data["train/grad_norm"] = grad_norm_value
                log_data["train/logit_scale"] = model.logit_scale.exp().item()
                if model.logit_bias is not None:
                    log_data["train/logit_bias"] = model.logit_bias.item()
                wandb_run.log(log_data, step=global_step)

            if (epoch <= args.lr_warmup_epochs - 1) and (warmup_lr_scheduler is not None):
                # still in the warmup phase
                warmup_lr_scheduler.step()
            else:
                if args.scheduler == "cosine_annealing":
                    main_lr_scheduler.step(
                        epoch - args.lr_warmup_epochs + update_step / num_updates_per_epoch
                    )  # pyright: ignore[reportArgumentType]
                else:
                    main_lr_scheduler.step()

            if (update_step + 1) % args.log_file_interval == 0:
                memory_suffix = ""
                if device.type == "cuda":
                    memory_stats = utils.get_gpu_memory_stats(device)
                    memory_suffix = (
                        f"\tmem_alloc={memory_stats['allocated_mib']:.1f}"
                        f"({memory_stats['peak_allocated_mib']:.1f} peak) MiB"
                        f"\tmem_reserved={memory_stats['reserved_mib']:.1f}"
                        f"({memory_stats['peak_reserved_mib']:.1f} peak) MiB"
                        f"\tnvidia-smi={memory_stats['nvml_used_mib']:.1f} MiB"
                    )
                loss_components_str = "".join(
                    f"\t{name}: {value:0.4f}" for name, value in detailed_loss_components.items()
                )
                logger.info(
                    f"Train: [{epoch + 1}/{args.num_epochs}][{update_step + 1}/{num_updates_per_epoch}]\t"
                    f"loss: {batch_loss:0.4f}"
                    f"{loss_components_str}\t"
                    f"grad_norm: {grad_norm_value:0.4f}\t"
                    f"logit_scale: {model.logit_scale.exp().item():0.4f}"
                    f"{memory_suffix}"
                )

            train_loss.update(batch_loss, num_items_in_batch)
            train_progressbar.set_postfix(
                {
                    "loss": f"{batch_loss:0.4f}",
                    **{name: f"{value:0.4f}" for name, value in detailed_loss_components.items()},
                    "grad_norm": f"{grad_norm_value:0.4f}",
                }
            )
            global_step += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_training_elapsed = time.perf_counter() - epoch_training_start_time
        total_training_only_time += epoch_training_elapsed
        total_training_pairs += epoch_training_pairs

        epoch_pairs_per_sec = epoch_training_pairs / epoch_training_elapsed
        logger.info(
            f"Training epoch {epoch + 1} time: {utils.to_hms(epoch_training_elapsed)}\t"
            f"image-text pairs/s: {epoch_pairs_per_sec:0.1f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_loss": train_loss.avg,
                    "train/epoch_time_seconds": epoch_training_elapsed,
                    "train/epoch_image_text_pairs_per_second": epoch_pairs_per_sec,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        # validation
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eval_start_time = time.perf_counter()
        val_results = eval_model(
            model=model,
            criterion=eval_criterion,
            eval_data_loader=val_data_loader,
            device=device,
        )
        print_eval_results(eval_results=val_results, prefix="val", epoch=epoch + 1, logger=logger)

        maybe_log_eval_results(
            eval_results=val_results,
            epoch=epoch,
            prefix="val",
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_eval_time += time.perf_counter() - eval_start_time

        # testing (excluded from timing)
        test_results = eval_model(
            model=model,
            criterion=eval_criterion,
            eval_data_loader=test_data_loader,
            device=device,
        )
        print_eval_results(
            eval_results=test_results, prefix="test", epoch=epoch + 1, logger=logger
        )

        maybe_log_eval_results(
            eval_results=test_results,
            epoch=epoch,
            prefix="test",
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )

        # saving checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"model_epoch_{epoch + 1}.pth",
        )
        if model_bundle.save_trainable_state_only:
            model_state_dict = model.get_checkpoint_state_dict()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            model_state_dict = model.state_dict()
        state_dict_to_save = {
            "model_state_dict": model_state_dict,
            "val_results": val_results,
            "epoch": epoch,
            "global_step": global_step,
        }
        if model_bundle.uses_checkpoint_metadata:
            state_dict_to_save["model_metadata"] = model.get_checkpoint_metadata()  # pyright: ignore[reportAttributeAccessIssue]
        if not args.save_best_k_only:
            torch.save(state_dict_to_save, checkpoint_path)
        save_top_k_checkpoints(
            criterion_metrics=args.best_checkpoint_metrics,
            top_k=args.save_best_k,
            val_results=val_results,
            best_val_results=best_val_results,
            state_dict_to_save=state_dict_to_save,
            checkpoint_path_template=os.path.join(
                checkpoint_dir, f"model_epoch_{epoch + 1}_{{metric}}_{{metric_value:.4f}}.pth"
            ),
        )

        if early_stopping.early_stop(val_loss=val_results["loss"]):
            logger.info(
                f"Early stopped. No improvement in validation loss for "
                f"{early_stopping.patience} consecutive epochs."
            )
            break

    logger.add(
        sys.stdout,
        format=logger_init_config["fmt"],
        level=logger_init_config["level"],
    )
    logger.info(f"Training-only time: {utils.to_hms(total_training_only_time)}")
    logger.info(f"Evaluation (validation) time: {utils.to_hms(total_eval_time)}")
    logger.info(
        f"Total time (training + validation): "
        f"{utils.to_hms(total_training_only_time + total_eval_time)}"
    )
    if total_training_only_time > 0:
        logger.info(
            f"  Training throughput: "
            f"{total_training_pairs / total_training_only_time:0.1f} image-text pairs/s"
        )
    if device.type == "cuda":
        memory_stats = utils.get_gpu_memory_stats(device)
        logger.info(
            f"  Peak GPU memory: allocated={memory_stats['peak_allocated_mib']:.1f} MiB,"
            f" reserved={memory_stats['peak_reserved_mib']:.1f} MiB,"
            f" nvidia-smi={memory_stats['nvml_used_mib']:.1f} MiB"
        )

    if len(best_val_results.keys()) == 1:
        best_metric_key = list(best_val_results.keys())[0]
        best_metric_value, best_checkpoint_path = sorted(
            best_val_results[best_metric_key], reverse=True
        )[0]
        logger.info(
            f"Best checkpoint based on {best_metric_key}: {best_checkpoint_path} "
            f"with value {best_metric_value:0.6f}"
        )
        logger.info("Loading best checkpoint for final evaluation on test set...")

        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        load_model_checkpoint_state(
            checkpoint["model_state_dict"],
            checkpoint.get("model_metadata"),
        )

        _run_test_only(test_data_loader)


def main():
    parser = argparse.ArgumentParser(
        description="Training model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_training_opts(parser)
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
