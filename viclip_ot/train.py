import argparse
import os
from datetime import datetime

import viclip_ot.utils as utils
from viclip_ot.opts import add_training_opts
from viclip_ot.utils.logger import init_logger, logger


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

    _logger_init_config = init_logger(level="DEBUG", log_file=log_file, compact=True)
    utils.set_seed(args.seed)

    logger.info(42)


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
