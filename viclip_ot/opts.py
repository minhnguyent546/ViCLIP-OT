import argparse
import os


def add_training_opts(parser: argparse.ArgumentParser) -> None:
    """
    All options used for training the model.
    """
    _add_model_and_dataset_opts(parser, is_training=True)
    _add_training_opts(parser)
    _add_wandb_opts(parser)


def add_training_with_co_teaching_opts(parser: argparse.ArgumentParser) -> None:
    """
    All options used for training the model with Co-Teaching.
    """
    _add_model_and_dataset_opts(parser)
    _add_training_opts(parser)
    _add_co_teaching_opts(parser)
    _add_wandb_opts(parser)


def add_test_with_model_soups_opts(parser: argparse.ArgumentParser) -> None:
    _add_model_and_dataset_opts(parser)

    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        help=(
            "Can be either a checkpoint file (.pth file) or a directory. "
            "In case of a directory, all of the checkpoints in that directory "
            "will be evaluated."
        ),
        required=True,
    )
    parser.add_argument(
        "--remove_duplicate_checkpoints",
        action="store_true",
        help="Whether to remove duplicate checkpoints (checkpoints saved for the same epoch but with different metrics)",
    )
    parser.add_argument(
        "--uniform_soup",
        action="store_true",
        help="Whether to compute uniform soup",
    )
    parser.add_argument(
        "--greedy_soup",
        action="store_true",
        help="Whether to compute greedy soup",
    )
    parser.add_argument(
        "--pruned_soup",
        action="store_true",
        help="Whether to compute pruned soup",
    )
    parser.add_argument(
        "--pruned_soup_num_iters",
        type=int,
        help="Number of iterations for pruned soup",
        default=8,
    )
    parser.add_argument(
        "--greedy_soup_comparison_metric",
        type=str,
        choices=["accuracy", "precision", "recall", "f1", "loss"],
        help="Metric to use as the comparison metric for greedy soup and pruned soup. `f1` is recommended as it usually has better generalization, reducing bias between validation and test sets.",
        default="f1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the results",
        default="./soups_results",
    )


def add_test_multiple_checkpoints_opts(parser: argparse.ArgumentParser) -> None:
    _add_model_and_dataset_opts(parser)

    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        help=(
            "Can be either a checkpoint file (.pth file) or a directory. "
            "In case of a directory, all of the checkpoints in that directory "
            "will be evaluated."
        ),
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File to save the evaluation results (.json file)",
        default="./test_results.json",
    )


def add_visualize_with_mds_opts(parser: argparse.ArgumentParser) -> None:
    _add_model_and_dataset_opts(parser)

    parser.add_argument(
        "--use_landmark_mds",
        action="store_true",
        help="Whether to use landmark MDS",
    )
    parser.add_argument(
        "--greedy_soup_checkpoint",
        type=str,
        help="Path to the greedy soup checkpoint file (.pth file)",
        default=None,
    )
    parser.add_argument(
        "--uniform_soup_checkpoint",
        type=str,
        help="Path to the uniform soup checkpoint file (.pth file)",
        default=None,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        nargs="+",
        help=(
            "Can be either a checkpoint file (.pth file) or a directory. "
            "In case of a directory, all of the checkpoints in that directory "
            "will be evaluated."
        ),
        required=True,
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "val", "test"],
        help="Which split will be used for visualization",
        default="val",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File to save the visualization results (e.g., .png, .pdf)",
        default="./vis_result.pdf",
    )


def add_self_influence_opts(parser: argparse.ArgumentParser) -> None:
    _add_model_and_dataset_opts(parser)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        nargs="+",
        help=(
            "Can be either a checkpoint file (.pth file) or a directory. "
            "In case of a directory, all of the checkpoints in that directory "
            "will be evaluated."
        ),
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file where results will be saved",
        default="self_influence_scores.json",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(os.cpu_count() or 1, 16),  # too large can cause insufficient shared memory
        help="Number of workers for data loading",
    )


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
        "--model",
        type=str,
        help="Name of the model to use (e.g., resnet50, densenet121, timm/coatnet_0_rw_224.sw_in1k, timm/maxvit_base_tf_224.in1k)",
        default="timm/coatnet_0_rw_224.sw_in1k",
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
        default="./data/ich-17",
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
        "--num_epochs",
        type=int,
        help="Number of training epochs",
        default=10,
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
        default=min(os.cpu_count() or 1, 16),  # too large can cause insufficient shared memory
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
        "--lr",
        type=float,
        help="Learning rate",
        default=1.0e-4,
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

    # EMA
    group.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average (EMA) for the model",
    )
    group.add_argument(
        "--model_ema_decay",
        type=float,
        help="Decay factor for Model EMA",
        default=0.9999,
    )
    group.add_argument(
        "--model_ema_warmup",
        action="store_true",
        help="Whether to use warmup for Model EMA",
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
        choices=["accuracy", "precision", "recall", "f1", "loss"],
        help="Metric to use for saving the best checkpoint (based on validation results)",
    )
    group.add_argument(
        "--save_best_k",
        type=int,
        help="Save upto `save_best_k` best checkpoints (do not use too large value as it can create a bottleneck in the training loop, recommended value is <= 5)",
        default=1,
    )

    # other
    group.add_argument(
        "--class_weighting",
        action="store_true",
        help="Whether to use class weighting for the training dataset via `WeightedRandomSampler`",
    )
    group.add_argument(
        "--use_mixup_cutmix",
        action="store_true",
        help="Whether to use MixUp & CutMiX",
    )
    group.add_argument(
        "--mixup_alpha",
        type=float,
        help="MixUp alpha (recommended values are 0.1 to 0.4)",
        default=0.2,
    )
    group.add_argument(
        "--cutmix_alpha",
        type=float,
        help="CutMiX alpha",
        default=1.0,
    )
    group.add_argument(
        "--max_grad_norm",
        type=float,
        help="Maximum gradient norm for gradient clipping",
        default=0.0,
    )


def _add_co_teaching_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Co-Teaching")
    group.add_argument(
        "--forget_rate",
        type=float,
        help="Forget rate for Co-Teaching",
        default=0.2,
    )
    group.add_argument(
        "--num_gradual_epochs",
        type=int,
        help="How many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.",
        default=10,
    )
    group.add_argument(
        "--forget_rate_exponent",
        type=float,
        help="Exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.",
        default=1,
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
        default="soups",
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
