import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms.v2 as v2
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import viclip_ot.constants as C
import viclip_ot.losses as losses
import viclip_ot.utils as utils
from viclip_ot.dataset import ImageTextCollate, ImageTextDataset
from viclip_ot.model import ViCLIPOT, ViCLIPOTConfig
from viclip_ot.opts import add_training_opts
from viclip_ot.utils.logger import init_logger, logger
from viclip_ot.utils.metric import AverageMeter
from viclip_ot.utils.training import (
    EarlyStopping,
    eval_model,
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

    logger_init_config = init_logger(level="DEBUG", log_file=log_file, compact=True)
    utils.set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Args: {args}")
    logger.info(f"Effective batch size: {args.train_batch_size * args.gradient_accum_steps}")

    # training device
    device = utils.get_device(args.device)
    logger.info(f"Using device: {device}")

    # creating model

    model_config = ViCLIPOTConfig.model_validate(utils.load_yaml_file(args.model_config))
    logger.info(f"Model config: {model_config}")
    model = ViCLIPOT(config=model_config)
    tokenizer = model.text_encoder.tokenizer
    logger.info(f"Model: {model}")
    model.to(device)

    if args.linear_probing:
        raise NotImplementedError("Loading from checkpoint is not implemented yet.")
        logger.info("Linear probing enabled")

    if args.from_checkpoint is not None:
        raise NotImplementedError("Loading from checkpoint is not implemented yet.")
        logger.info(f"Loading model from checkpoint: {args.from_checkpoint}")
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # loading dataset
    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=args.train_crop_size, interpolation=v2.InterpolationMode.BICUBIC
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.ToTensor(),
            v2.Normalize(
                mean=C.IMAGENET_DEFAULT_MEAN,
                std=C.IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    eval_transforms = v2.Compose(
        [
            v2.Resize(size=args.eval_resize_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size=args.eval_crop_size),
            v2.ToTensor(),
            v2.Normalize(
                mean=C.IMAGENET_DEFAULT_MEAN,
                std=C.IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    train_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file="train.json",
        image_transforms=train_transforms,
    )
    test_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file="test.json",
        image_transforms=eval_transforms,
    )
    val_dataset = ImageTextDataset(
        root_dir=args.dataset_dir,
        metadata_json_file="val.json",
        image_transforms=eval_transforms,
    )

    logger.info(
        f"train_size = {len(train_dataset)}, "
        f"test_size = {len(test_dataset)}, "
        f"val_size = {len(val_dataset)} "
    )

    collate_fun = ImageTextCollate(tokenizer=tokenizer)
    # creating data loaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fun,
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fun,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fun,
        persistent_workers=True,
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
            config=vars(args),
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
    logger.info(f"num_params: {utils.to_human_readable(num_model_params)}")

    if args.run_test_only:
        raise NotImplementedError("Test only mode is not implemented yet.")
        return

    assert checkpoint_dir is not None

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
        main_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr,
            pct_start=0.0,  # warmup is now can be used via a separate scheduler
            epochs=args.num_epochs - args.lr_warmup_epochs,
            steps_per_epoch=num_updates_per_epoch,
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
    if args.best_checkpoint_metrics is None:
        args.best_checkpoint_metrics = []

    # best_val_results[metric] = list of tuples (value, checkpoint_path)
    best_val_results: dict[str, list[tuple[float, str]]] = {
        metric: [] for metric in args.best_checkpoint_metrics
    }
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, min_delta=0.0, enabled=args.early_stopping
    )
    if early_stopping.is_enabled():
        logger.info(f"Early stopping enabled with patience {early_stopping.patience}")

    # disable logging to stdout during training to avoid conflict with tqdm
    logger.remove(logger_init_config["stdout_id"])

    global_step = 0
    training_start_time = time.perf_counter()
    criterion = losses.ClipLoss(cache_labels=True)
    for epoch in range(args.num_epochs):
        model.train()

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
            for batch in batches:
                images = batch["images"].to(device=device, non_blocking=True)
                text_inputs = batch["text_inputs"].to(device=device, non_blocking=True)

                with autocast_context:
                    model_outputs = model(images, text_inputs)
                    loss = criterion(
                        image_features=model_outputs["image_features"],
                        text_features=model_outputs["text_features"],
                        logit_scale=model_outputs["logit_scale"],
                        logit_bias=model_outputs.get("logit_bias", None),
                        reduction="sum",
                    )
                    if num_items_in_batch > 0:
                        loss = loss / num_items_in_batch

                scaler.scale(loss).backward()
                batch_loss += loss.detach().item()

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

            # note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                model.logit_scale.clamp_(min=0, max=np.log(100))

            if wandb_run is not None:
                log_data = {
                    f"learning_rate/group_{group_id}": param_group["lr"]
                    for group_id, param_group in enumerate(optimizer.param_groups)
                }
                log_data["train/loss"] = batch_loss
                log_data["train/grad_norm"] = grad_norm_value
                log_data["train/logit_scale"] = model.logit_scale.exp().item()
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
                memory_used = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                    if device.type == "cuda"
                    else 0
                )
                logger.info(
                    f"Train: [{epoch + 1}/{args.num_epochs}][{update_step + 1}/{num_updates_per_epoch}]\t"
                    f"loss: {batch_loss:0.4f}\t"
                    f"grad_norm: {grad_norm_value:0.4f}\t"
                    f"logit_scale: {model.logit_scale.exp().item():0.4f}\t"
                    f"memory: {memory_used:0.2f} MB"
                )

            train_loss.update(batch_loss, num_items_in_batch)
            train_progressbar.set_postfix(
                {
                    "loss": f"{batch_loss:0.4f}",
                    "grad_norm": f"{grad_norm_value:0.4f}",
                }
            )
            global_step += 1

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_loss": train_loss.avg,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        # validation
        val_results = eval_model(
            model=model,
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

        # testing
        test_results = eval_model(
            model=model,
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
        state_dict_to_save = {
            "model_state_dict": model.state_dict(),
            "val_results": val_results,
            "epoch": epoch,
            "global_step": global_step,
        }
        torch.save(state_dict_to_save, checkpoint_path)
        save_top_k_checkpoints(
            criterion_metrics=args.best_checkpoint_metrics,
            top_k=args.save_best_k,
            val_results=val_results,
            best_val_results=best_val_results,
            state_dict_to_save=state_dict_to_save,
            checkpoint_path_template=os.path.join(
                checkpoint_dir, f"model_epoch_{epoch}_{{metric}}_{{metric_value:.4f}}.pth"
            ),
        )

        if early_stopping.early_stop(val_loss=val_results["loss"]):
            logger.info(
                f"Early stopped. No improvement in validation loss for "
                f"{early_stopping.patience} consecutive epochs."
            )
            break

    total_training_time = time.perf_counter() - training_start_time
    logger.info(f"Training time: {utils.to_hms(total_training_time)}")


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
