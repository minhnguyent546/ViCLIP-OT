import argparse
import os


def _float_or_none(value: str) -> float | None:
    """Parse a CLI value into a float, mapping 'none'/'null' to None."""
    if value.strip().lower() in {"none", "null"}:
        return None
    return float(value)


def add_training_opts(parser: argparse.ArgumentParser) -> None:
    """
    All options used for training the model.
    """
    _add_model_and_dataset_opts(parser, is_training=True)
    _add_training_opts(parser)
    _add_wandb_opts(parser)


def _add_model_and_dataset_opts(
    parser: argparse.ArgumentParser, is_training: bool = False
) -> None:
    group = parser.add_argument_group("Model & Dataset")
    group.add_argument(
        "--seed",
        type=int,
        help="Seed for random number generators",
        default=42,
    )
    group.add_argument(
        "--model_config",
        type=str,
        help="Path to the model config file",
        default="./config/model.yaml",
    )
    group.add_argument(
        "--device",
        type=str,
        help="Which device to use (e.g., cpu, cuda, cuda:7, auto)",
        default="auto",
    )
    group.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the dataset",
        default="./data/UIT-OpenViIC",
    )
    group.add_argument(
        "--train_split_name",
        type=str,
        help="Training split name",
        default="train",
    )
    group.add_argument(
        "--test_split_name",
        type=str,
        help="Testing split name",
        default="test",
    )
    group.add_argument(
        "--val_split_name",
        type=str,
        help="Validation split name",
        default="val",
    )
    if is_training:
        group.add_argument(
            "--train_batch_size",
            type=int,
            help="Training batch size",
            default=32,
        )
        group.add_argument(
            "--train_crop_size",
            type=int,
            help="Random crop size used for training",
            default=224,
        )
    group.add_argument(
        "--eval_batch_size",
        type=int,
        help="Evaluation batch size",
        default=32,
    )
    group.add_argument(
        "--eval_resize_size",
        type=int,
        help="Resize size used for evaluation",
        default=256,
    )
    group.add_argument(
        "--eval_crop_size",
        type=int,
        help="Central crop size used for evaluation",
        default=224,
    )


def _add_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Training")

    # test only mode
    group.add_argument(
        "--run_test_only",
        action="store_true",
        help="Run testing only",
    )

    # basic stuff
    group.add_argument(
        "--checkpoints_dir",
        type=str,
        help="Checkpoints directory for saving stuff",
        default="./checkpoints",
    )
    group.add_argument(
        "--from_checkpoint",
        type=str,
        help="Path to the checkpoint storing the model state",
    )
    group.add_argument(
        "--random_weights",
        action="store_true",
        help="Whether to initializing models with random weights instead of initializing with pretrained weights (this option takes no effect when `from_checkpoint` is specified)",
    )
    group.add_argument(
        "--linear_probing",
        action="store_true",
        help="Whether to perform linear probing (only train the final classifier layer)",
    )
    group.add_argument(
        "--criterion",
        type=str,
        help="Which loss criterion to use",
        choices=[
            "clip_loss",
            "sig_lip_loss",
            "batch_level_entropic_ot_loss",
            "hybrid_clip_tp_loss",
            "hybrid_sig_lip_tp_loss",
        ],
        default="clip_loss",
    )
    group.add_argument(
        "--use_transport_plan_as_logits",
        action="store_true",
        help="Whether to use the transport plan as logits for computing cross-entropy loss in batch-level entropic OT loss, otherwise the transport plan is used as soft targets. Note that these two cases will only happen when `--sim_graph_regularized_ot` is not set.",
    )
    group.add_argument(
        "--sim_matrix_temperature",
        type=_float_or_none,
        help="Temperature for scaling the similarity matrix when computing OT losses. Use `--sim_matrix_temperature none` for no scaling.",
        default=None,
    )
    group.add_argument(
        "--sinkhorn_solver",
        type=str,
        help="Sinkhorn solver to use for OT losses",
        choices=[
            "sinkhorn",
            "sinkhorn_unbalanced",
        ],
        default="sinkhorn_unbalanced",
    )
    group.add_argument(
        "--sim_graph_regularized_ot",
        action="store_true",
        help="Whether to use similarity graph regularized OT for batch-level entropic OT loss",
    )
    group.add_argument(
        "--precomputed_caption_embeddings_path",
        type=str,
        help="Path to precomputed caption embeddings for similarity graph regularized OT (.pt file)",
    )
    group.add_argument(
        "--precomputed_image_embeddings_path",
        type=str,
        help="Path to precomputed image embeddings for similarity graph regularized OT (.pt file)",
    )
    group.add_argument(
        "--sim_graph_alpha",
        type=float,
        help="Weight for image embeddings for similarity graph regularized OT. The weight for text embeddings is (1 - sim_graph_alpha)",
        default=0.5,
    )
    group.add_argument(
        "--sim_combine_method",
        type=str,
        help="Method to combine similarity graphs from image and text modalities",
        choices=[
            "weighted_sum",
            "maximum",
            "harmonic_mean",
            "sparse_thresholding",
            "minimum",
            "power_mean",
            "arithmetic_mean",
            "cross_modality",
            "cross_modality3",
        ],
        default="cross_modality",
    )
    group.add_argument(
        "--sim_power_mean_exponent",
        type=float,
        help="Exponent p for power mean when using 'power_mean' as sim_combine_method",
        default=3.0,
    )
    group.add_argument(
        "--sim_sparse_threshold_quantile",
        type=float,
        help="Quantile value (between 0 and 1) for sparse thresholding when using 'sparse_thresholding' as sim_combine_method",
        default=0.5,
    )
    group.add_argument(
        "--hybrid_clip_tp_loss_clip_loss_lambda",
        type=float,
        help="Lambda weight for CLIP loss in hybrid CLIP and TP loss (loss = lambda * CLIP_loss + OT_loss)",
        default=0.1,
    )
    group.add_argument(
        "--hybrid_sig_lip_tp_loss_sig_lip_loss_lambda",
        type=float,
        help="Lambda weight for SigLIP loss in hybrid SigLIP and TP loss (loss = lambda * SigLIP_loss + OT_loss)",
        default=0.1,
    )

    group.add_argument(
        "--num_epochs",
        type=int,
        help="Number of training epochs",
        default=30,
    )
    group.add_argument(
        "--label_smoothing",
        type=float,
        help="Label smoothing value",
        default=0.0,
    )
    group.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for data loading",
        default=min(
            max((os.cpu_count() or 1) // 2, 1), 4
        ),  # too large can cause insufficient shared memory
    )
    group.add_argument(
        "--log_file_interval",
        type=int,
        help="Interval (in steps) for logging training progress",
        default=10,
    )

    # mixed precision training
    group.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16"],
        help="Whether to enable mixed precision training (fp16 or bf16)",
    )

    # gradient accumulation
    group.add_argument(
        "--gradient_accum_steps",
        type=int,
        help="Number of gradient accumulation steps",
        default=1,
    )

    # optimizer
    group.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to use",
        choices=["adam", "adamw"],
        default="adamw",
    )
    group.add_argument(
        "--adam_betas",
        type=float,
        nargs=2,
        help="Betas for Adam/AdamW optimizer. Pass it as --adam_betas <beta1> <beta2>",
        default=(0.9, 0.999),
    )
    group.add_argument(
        "--adam_eps",
        type=float,
        help="Epsilon for Adam/AdamW optimizer",
        default=1e-8,
    )
    group.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=1.0e-4,
    )
    group.add_argument(
        "--backbone_lr",
        type=float,
        help="Learning rate for backbone parameters, should be set lower than or equal to `--lr`",
        default=1.0e-5,
    )
    group.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay",
        default=1e-4,
    )

    # scheduler
    group.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine_annealing", "one_cycle_lr"],
        help="Which learning rate scheduler to use",
        default="cosine_annealing",
    )
    group.add_argument(
        "--min_lr",
        type=float,
        help="Minimum learning rate",
        default=0.0,
    )
    group.add_argument(
        "--lr_warmup_epochs",
        type=int,
        help="Number of epochs to warmup",
        default=0,
    )
    group.add_argument(
        "--lr_warmup_method",
        type=str,
        choices=["linear", "constant"],
        help="Learning rate warmup method",
        default="linear",
    )
    group.add_argument(
        "--lr_warmup_decay",
        type=float,
        help="Decay for learning rate",
        default=0.01,
    )
    group.add_argument(
        "--cosine_annealing_T_0",
        type=int,
        help="cosine_annealing: Number of iterations for the first restart",
        default=10,
    )
    group.add_argument(
        "--cosine_annealing_T_mult",
        type=int,
        help="cosine_annealing: Multiplier for the period of the cosine annealing scheduler",
        default=3,
    )

    # LiT
    group.add_argument(
        "--lock_image",
        action="store_true",
        help="Whether to lock the (trunk in) image_encoder",
    )
    group.add_argument(
        "--lock_text",
        action="store_true",
        help="Whether to lock the (encoder in) text_encoder",
    )
    group.add_argument(
        "--lock_image_last_unfreeze_groups",
        type=int,
        help="Leave last n layer groups in image_encoder unlocked",
        default=0,
    )
    group.add_argument(
        "--lock_image_freeze_bn_stats",
        action="store_true",
        help="Whether to freeze BatchNorm running stats for any locked layers",
    )
    group.add_argument(
        "--lock_text_unfreeze_dense",
        action="store_true",
        help="Whether to leave dense layers in text_encoder unlocked",
    )

    # early stopping
    group.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to use early stopping",
    )
    group.add_argument(
        "--early_stopping_patience",
        type=int,
        help="Patience for early stopping",
        default=5,
    )
    # save best checkpoints
    group.add_argument(
        "--best_checkpoint_metrics",
        type=str,
        nargs="*",
        choices=[
            "_loss",
            "_i2t_mean_rank",
            "_i2t_median_rank",
            "i2t_R__1",
            "i2t_R__5",
            "i2t_R__10",
            "_t2i_mean_rank",
            "_t2i_median_rank",
            "t2i_R__1",
            "t2i_R__5",
            "t2i_R__10",
        ],
        help="Metric to use for saving the best checkpoint (based on validation results). Prefix with '_' to indicate decreasing order (e.g. _loss).",
        default=["_loss"],
    )
    group.add_argument(
        "--save_best_k",
        type=int,
        help="Save upto `save_best_k` best checkpoints (do not use too large value as it can create a bottleneck in the training loop, recommended value is <= 5)",
        default=5,
    )
    group.add_argument(
        "--save_best_k_only",
        action="store_true",
        help="Save only best k checkpoints",
    )

    # other
    group.add_argument(
        "--max_grad_norm",
        type=float,
        help="Maximum gradient norm for gradient clipping",
        default=0.0,
    )


def _add_wandb_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Wandb")
    group.add_argument(
        "--wandb_logging",
        action="store_true",
        help="Enable logging to wandb",
    )
    group.add_argument(
        "--wandb_project",
        type=str,
        help="Project name",
        default="viclip_ot",
    )
    group.add_argument(
        "--wandb_name",
        type=str,
        help="Experiment name",
    )
    group.add_argument(
        "--wandb_resume_id",
        type=str,
        help="Id to resume a run from",
    )
    group.add_argument(
        "--wandb_notes",
        type=str,
        help="Wandb notes",
    )
    group.add_argument(
        "--wandb_tags",
        type=str,
        help="Wandb tags",
    )
