import argparse
import os


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
