# pyright: reportPrivateImportUsage=false

import contextlib
from typing import Literal

import ot
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch import Tensor

from viclip_ot.utils.logger import logger


def _sinkhorn(
    metric_cost_matrix: Tensor,
    reg: float = 0.05,  # try with [0.01, 0.1] with normalized cost matrix
    max_num_iters: int = 200,
) -> Tensor:
    """
    Solve the entropic regularization optimal transport problem and return the OT plan.

    Reference: https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn
    """
    device = metric_cost_matrix.device
    batch_size = metric_cost_matrix.shape[0]
    a = torch.ones((batch_size,), device=device) / batch_size
    b = torch.ones((batch_size,), device=device) / batch_size

    transport_plan = ot.sinkhorn(
        a=a,
        b=b,
        M=metric_cost_matrix,
        reg=reg,
        method="sinkhorn_log",
        numItermax=max_num_iters,
    )

    # scale with batch_size as we initialize `a` and `b` with equal weight `1 / N` for each image (and text)
    return transport_plan * batch_size  # pyright: ignore[reportReturnType]


def _sinkhorn_unbalanced(
    metric_cost_matrix: Tensor,
    reg: float = 0.05,  # try with [0.01, 0.1] with normalized cost matrix
    reg_m: float = 0.5,  # try with [0.3, 0.8] with normalized cost matrix
    max_num_iters: int = 200,
):
    """
    Solve the entropic regularization unbalanced optimal transport problem and return the OT plan.

    Reference: https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.sinkhorn_unbalanced

    """
    device = metric_cost_matrix.device
    batch_size = metric_cost_matrix.shape[0]

    a = torch.ones(batch_size, device=device) / batch_size
    b = torch.ones(batch_size, device=device) / batch_size

    # This solves the transport where rows/cols don't have to sum exactly to 1/N
    transport_plan = ot.sinkhorn_unbalanced(
        a,
        b,
        metric_cost_matrix,
        reg=reg,
        reg_m=reg_m,
        reg_type="kl",
        method="sinkhorn_stabilized",
        numItermax=max_num_iters,
    )

    # scale with batch_size as we initialize `a` and `b` with equal weight `1 / N` for each image (and text)
    return transport_plan * batch_size  # pyright: ignore[reportReturnType]


def _compute_logits(
    image_features: Tensor,
    text_features: Tensor,
    logit_scale: Tensor | None = None,
    logit_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled cosine similarity logits for image-text pairs.

    Computes in float32 for numerical stability with mixed precision training,
    then casts back to the original dtype.
    """
    original_dtype = image_features.dtype
    logits_per_image = image_features.float() @ text_features.float().t()
    if logit_scale is not None:
        logits_per_image = logit_scale * logits_per_image

    logits_per_text = logits_per_image.t()

    if logit_bias is not None:
        logits_per_image = logits_per_image + logit_bias
        logits_per_text = logits_per_text + logit_bias

    return logits_per_image.to(original_dtype), logits_per_text.to(original_dtype)


def _reduce_sample_losses(sample_losses: Tensor, reduction: str) -> Tensor:
    """Reduce a vector of per-sample losses with CLIP/CE-compatible semantics."""
    if reduction == "mean":
        return sample_losses.mean()
    if reduction == "sum":
        return sample_losses.sum()
    if reduction == "none":
        return sample_losses
    raise ValueError(
        f"Unsupported reduction: {reduction}. Expected one of ['mean', 'sum', 'none']."
    )


def _generalized_kl(P: Tensor, Q: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute generalized KL divergence between true probability distribution ``P`` and
    approximated probability distribution ``Q``.
    """
    divergence = P * (P.clamp_min(eps).log() - Q.clamp_min(eps).log() - 1) + Q
    return divergence


class ClipLoss(nn.Module):
    """(Open) CLIP loss.

    Taken and modified from: https://github.com/mlfoundations/open_clip/blob/d3cdb734a2710feeb4c6307df037afa5f786a3e1/src/open_clip/loss.py#L68.

    Modification:
    - Add type hints.
    - Removed multi-node support for simplicity.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        image_ids: Tensor | None = None,
        output_dict: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        """
        If `image_ids` is provided, the computation will take into account image with multiple captions.
        """
        device = image_features.device
        logits_per_image, logits_per_text = _compute_logits(
            image_features,
            text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

        if image_ids is None:
            # 1-1 matching between image and text
            labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.int64)

            loss_i2t = Fun.cross_entropy(
                input=logits_per_image, target=labels, reduction=reduction
            )
            loss_t2i = Fun.cross_entropy(input=logits_per_text, target=labels, reduction=reduction)
        else:
            image_ids = image_ids.to(device)
            # shape: (batch_size, batch_size)
            matches = image_ids.view(-1, 1) == image_ids.view(1, -1)

            # create soft labels (probs)
            soft_labels = matches.float()
            soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
            loss_i2t = Fun.cross_entropy(logits_per_image, soft_labels, reduction=reduction)
            loss_t2i = Fun.cross_entropy(logits_per_text, soft_labels, reduction=reduction)

        total_loss = (loss_i2t + loss_t2i) / 2

        return {"loss": total_loss} if output_dict else total_loss


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    Adapted from: https://github.com/mlfoundations/open_clip/blob/d3cdb734a2710feeb4c6307df037afa5f786a3e1/src/open_clip/loss.py#L330

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(self):
        super().__init__()

    def get_ground_truth(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_logits: int,
        negative_only: bool = False,
    ) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
    ):
        # shape: (batch_size, batch_size)
        original_dtype = image_features.dtype
        logits = logit_scale * image_features.float() @ text_features.float().T
        if logit_bias is not None:
            logits += logit_bias

        return logits.to(original_dtype)

    def _loss(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        image_ids: Tensor | None = None,
        negative_only: bool = False,
        reduction: str = "mean",
    ) -> Tensor:
        # shape: (batch_size, batch_size)
        batch_size = image_features.shape[0]
        logits = self.get_logits(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )
        # shape: (batch_size, batch_size)
        if image_ids is None:
            labels = self.get_ground_truth(
                device=image_features.device,
                dtype=logits.dtype,
                num_logits=batch_size,
                negative_only=negative_only,
            )
            weights = None
        else:
            image_ids = image_ids.to(image_features.device)
            matches = image_ids.view(-1, 1) == image_ids.view(1, -1)
            if negative_only:
                labels = -torch.ones(
                    (batch_size, batch_size),
                    device=image_features.device,
                    dtype=logits.dtype,
                )
                weights = None
            else:
                labels = matches.to(dtype=logits.dtype) * 2 - 1
                num_pos = matches.sum(dim=1, keepdim=True).clamp_min(1)
                weights = torch.ones_like(labels)
                weights = (
                    weights.masked_fill(matches, 0.0) + matches.to(dtype=logits.dtype) / num_pos
                )

        loglik = Fun.logsigmoid(labels * logits)
        if weights is not None:
            loglik = loglik * weights
        nll = -loglik.sum(dim=-1)

        loss = nll
        if reduction == "sum":
            loss = nll.sum()
        elif reduction == "mean":
            loss = nll.mean()
        elif reduction == "none":
            pass

        return loss

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        image_ids: Tensor | None = None,
        output_dict: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        loss = self._loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            image_ids=image_ids,
            reduction=reduction,
        )

        return {"loss": loss} if output_dict else loss


class BatchLevelEntropicOTLoss(nn.Module):
    """
    Batch-level Entropic Optimal Transport Loss
    """

    def __init__(
        self,
        sinkhorn_solver: Literal["sinkhorn", "sinkhorn_unbalanced"] = "sinkhorn_unbalanced",
        use_transport_plan_as_logits: bool = False,
        sim_matrix_temperature: float = 0.05,
        sinkhorn_reg: float = 0.05,
        sinkhorn_reg_m: float = 0.5,
        sinkhorn_max_num_iters: int = 200,
        sigrot_unbalanced_variant: Literal[
            "raw_gkl",
            "row_norm_mass_weighted",
            "mass_matched_gkl",
        ] = "raw_gkl",
    ):
        """
        In case `sim_matrix` is not provided in the forward pass: If `use_transport_plan_as_logits` is True,
        the transport plan will be used as logits for computing the cross-entropy loss,
        otherwise, use transport plan as soft labels.

        If `sim_matrix` is provided in the forward pass, `use_transport_plan_as_logits` takes no effect.
        """

        super().__init__()
        self.sinkhorn_solver = sinkhorn_solver
        self.use_transport_plan_as_logits = use_transport_plan_as_logits
        self.sim_matrix_scale_factor = 1 / sim_matrix_temperature
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_reg_m = sinkhorn_reg_m
        self.sinkhorn_max_num_iters = sinkhorn_max_num_iters
        self.sigrot_unbalanced_variant = sigrot_unbalanced_variant
        self.last_transport_stats: dict[str, float | str] = {}

        logger.info(f"Using Sinkhorn Solver: {self.sinkhorn_solver}")
        logger.info(
            "Using Sinkhorn parameters: "
            f"reg={self.sinkhorn_reg}, "
            f"reg_m={self.sinkhorn_reg_m}, "
            f"max_num_iters={self.sinkhorn_max_num_iters}"
        )
        if self.sinkhorn_solver == "sinkhorn_unbalanced":
            logger.info(f"Using SIGROT unbalanced variant: {self.sigrot_unbalanced_variant}")

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        output_dict: bool = False,
        sim_matrix: Tensor | None = None,
        image_ids: Tensor | None = None,
        reduction: str = "mean",
        **kwargs,
    ):
        """
        If `self.use_transport_plan_as_logits` is True (i.e., use transport plan as logits)
        and `image_ids` is provided, the computation will take into account image with multiple captions,
        meaning that labels for cross-entropy loss is **soft labels** and will be created based on `image_ids`
        with equal weights for each caption.
        """
        batch_size = image_features.shape[0]

        # enable gradient computation only if `sim_matrix`` is provided or
        # `self.use_transport_plan_as_logits`` is True
        ctx = (
            contextlib.nullcontext()
            if (sim_matrix is not None or self.use_transport_plan_as_logits)
            else torch.no_grad()
        )
        with ctx:
            # values are in range [0, 2]
            cost_matrix_i2t = 1 - (image_features @ text_features.t())

            # compute transport plan `transport_plan_i2t` for the image -> text direction,
            # the transport plan for the reverse text -> image direction
            # is equivalent to `transport_plan_i2t.t()`
            if self.sinkhorn_solver == "sinkhorn":
                transport_plan_i2t = _sinkhorn(
                    metric_cost_matrix=cost_matrix_i2t,
                    reg=self.sinkhorn_reg,
                    max_num_iters=self.sinkhorn_max_num_iters,
                )
            elif self.sinkhorn_solver == "sinkhorn_unbalanced":
                transport_plan_i2t = _sinkhorn_unbalanced(
                    metric_cost_matrix=cost_matrix_i2t,
                    reg=self.sinkhorn_reg,
                    reg_m=self.sinkhorn_reg_m,
                    max_num_iters=self.sinkhorn_max_num_iters,
                )
            else:
                raise ValueError(f"Unsupported solver: {self.sinkhorn_solver}")

            transport_plan_i2t = transport_plan_i2t.clamp_min(0)
            transport_plan_t2i = transport_plan_i2t.t()

        self.last_transport_stats = {
            "variant": (
                "balanced"
                if self.sinkhorn_solver == "sinkhorn"
                else self.sigrot_unbalanced_variant
            )
        }

        if sim_matrix is not None:
            # SIGROT compares the student-induced OT structure against a teacher
            # similarity graph.  Balanced OT after `* batch_size` is already a
            # row/column probability matrix, but unbalanced OT carries extra
            # mass information.
            eps = torch.finfo(transport_plan_i2t.dtype).eps

            i2t_mass = transport_plan_i2t.sum(dim=1, keepdim=True)
            t2i_mass = transport_plan_t2i.sum(dim=1, keepdim=True)
            with torch.no_grad():
                self.last_transport_stats.update(  # pyright: ignore[reportCallIssue]
                    {
                        "i2t_mass_mean": i2t_mass.mean().item(),
                        "i2t_mass_std": i2t_mass.std(unbiased=False).item(),
                        "i2t_mass_min": i2t_mass.min().item(),
                        "i2t_mass_max": i2t_mass.max().item(),
                        "t2i_mass_mean": t2i_mass.mean().item(),
                        "t2i_mass_std": t2i_mass.std(unbiased=False).item(),
                        "t2i_mass_min": t2i_mass.min().item(),
                        "t2i_mass_max": t2i_mass.max().item(),
                    }  # pyright: ignore[reportArgumentType]
                )

            # Use independently row-normalized graph distributions for each
            # retrieval direction.  Note that softmax(S).T is not equivalent to
            # softmax(S.T), especially for cross-modal/asymmetric graph mixes.
            sim_i2t = (self.sim_matrix_scale_factor * sim_matrix).softmax(dim=1)
            sim_t2i = (self.sim_matrix_scale_factor * sim_matrix.t()).softmax(dim=1)

            if self.sinkhorn_solver == "sinkhorn" or self.sigrot_unbalanced_variant == "raw_gkl":
                # Baselines:
                # - `sinkhorn`: balanced OT, where N * P has row/column sums ~= 1,
                #   so GKL is effectively comparing compatible probability rows.
                # - `raw_gkl`: the previous unbalanced behavior. Here N * P is a
                #   non-normalized measure, while the teacher graph rows sum to 1.
                #   This intentionally keeps mass mismatch in the loss as an
                #   ablation to test whether raw unbalanced mass is useful or harmful.
                sample_loss_i2t = _generalized_kl(transport_plan_i2t, sim_i2t).sum(dim=1)
                sample_loss_t2i = _generalized_kl(transport_plan_t2i, sim_t2i).sum(dim=1)
                loss_i2t = _reduce_sample_losses(sample_loss_i2t, reduction)
                loss_t2i = _reduce_sample_losses(sample_loss_t2i, reduction)
            elif self.sigrot_unbalanced_variant == "row_norm_mass_weighted":
                # `row_norm_mass_weighted`: normalize the unbalanced plan into proper row-wise
                # distributions, then use detached row/column mass as relative confidence.
                #
                # Important: use mass / mean(mass), not raw mass. Raw unbalanced mass
                # also changes the global SIGROT loss scale as total transported mass
                # changes during training, which can make this branch much weaker than
                # the raw/balanced baselines. Normalizing preserves relative confidence
                # while keeping the average per-sample loss scale stable.
                plan_i2t = transport_plan_i2t / i2t_mass.clamp_min(eps)
                plan_t2i = transport_plan_t2i / t2i_mass.clamp_min(eps)

                row_kl_i2t = _generalized_kl(plan_i2t, sim_i2t).sum(dim=1)
                row_kl_t2i = _generalized_kl(plan_t2i, sim_t2i).sum(dim=1)

                weights_i2t = i2t_mass.detach() / i2t_mass.detach().mean().clamp_min(eps)
                weights_t2i = t2i_mass.detach() / t2i_mass.detach().mean().clamp_min(eps)
                weights_i2t = weights_i2t.clamp(min=0.5, max=2.0)
                weights_t2i = weights_t2i.clamp(min=0.5, max=2.0)
                # TODO: consider warmup/blending for these confidence weights, e.g.
                #   w = (1 - alpha) + alpha * clipped_normalized_mass,
                # with alpha scheduled from 0 to 1. This may avoid suppressing
                # currently-hard samples too aggressively early in training.
                sample_loss_i2t = weights_i2t.squeeze(1) * row_kl_i2t
                sample_loss_t2i = weights_t2i.squeeze(1) * row_kl_t2i
                loss_i2t = _reduce_sample_losses(sample_loss_i2t, reduction)
                loss_t2i = _reduce_sample_losses(sample_loss_t2i, reduction)
            elif self.sigrot_unbalanced_variant == "mass_matched_gkl":
                # `mass_matched_gkl`: keep the unbalanced plan as a measure, but scale the
                # teacher graph by detached OT mass so P and Q have compatible row
                # totals. This is measure-level GKL while avoiding the raw baseline's
                # accidental penalty against a fixed row-sum-1 teacher distribution.
                sim_measure_i2t = i2t_mass.detach() * sim_i2t
                sim_measure_t2i = t2i_mass.detach() * sim_t2i

                # Divide by mean detached mass for stable loss scale. This does not
                # remove the measure-level gradient behavior of this variant; it only
                # avoids making the effective SIGROT weight depend on total OT mass.
                sample_loss_i2t = _generalized_kl(transport_plan_i2t, sim_measure_i2t).sum(
                    dim=1
                ) / i2t_mass.detach().mean().clamp_min(eps)
                sample_loss_t2i = _generalized_kl(transport_plan_t2i, sim_measure_t2i).sum(
                    dim=1
                ) / t2i_mass.detach().mean().clamp_min(eps)
                loss_i2t = _reduce_sample_losses(sample_loss_i2t, reduction)
                loss_t2i = _reduce_sample_losses(sample_loss_t2i, reduction)
            else:
                raise ValueError(
                    f"Unsupported SIGROT unbalanced variant: {self.sigrot_unbalanced_variant}"
                )
        else:
            device = image_features.device
            if self.use_transport_plan_as_logits:
                # use transport plan as logits
                if image_ids is None:
                    # 1-1 matching between image and text
                    labels = torch.arange(batch_size, device=device, dtype=torch.int64)

                    loss_i2t = Fun.cross_entropy(
                        input=transport_plan_i2t, target=labels, reduction=reduction
                    )
                    loss_t2i = Fun.cross_entropy(
                        input=transport_plan_t2i, target=labels, reduction=reduction
                    )
                else:
                    image_ids = image_ids.to(device)
                    # shape: (batch_size, batch_size)
                    matches = image_ids.view(-1, 1) == image_ids.view(1, -1)

                    # create soft labels (probs)
                    soft_labels = matches.float()
                    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
                    loss_i2t = Fun.cross_entropy(
                        transport_plan_i2t, soft_labels, reduction=reduction
                    )
                    loss_t2i = Fun.cross_entropy(
                        transport_plan_t2i, soft_labels, reduction=reduction
                    )
            else:
                # use transport plan as soft targets

                logits_per_image, logits_per_text = _compute_logits(
                    image_features,
                    text_features,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                )

                loss_i2t = Fun.cross_entropy(
                    input=logits_per_image,
                    target=transport_plan_i2t,
                    reduction=reduction,
                )
                loss_t2i = Fun.cross_entropy(
                    input=logits_per_text,
                    target=transport_plan_t2i,
                    reduction=reduction,
                )

        total_loss = (loss_i2t + loss_t2i) / 2

        return {"loss": total_loss} if output_dict else total_loss


class HybridClipTPLoss(nn.Module):
    """
    Combines CLIP loss with Batch-level Entropic Optimal Transport Loss.
    """

    def __init__(
        self,
        clip_loss_lambda: float = 0.1,
        sinkhorn_solver: Literal["sinkhorn", "sinkhorn_unbalanced"] = "sinkhorn",
        use_transport_plan_as_logits: bool = False,
        sim_matrix_temperature: float = 0.05,
        sinkhorn_reg: float = 0.05,
        sinkhorn_reg_m: float = 0.5,
        sinkhorn_max_num_iters: int = 200,
        sigrot_unbalanced_variant: Literal[
            "raw_gkl",
            "row_norm_mass_weighted",
            "mass_matched_gkl",
        ] = "raw_gkl",
    ):
        """
        In case `sim_matrix` is not provided in the forward pass: If `use_transport_plan_as_logits` is True,
        the transport plan will be used as logits for computing the cross-entropy loss,
        otherwise, use transport plan as soft labels.

        If `sim_matrix` is provided in the forward pass, `use_transport_plan_as_logits` takes no effect.
        """

        super().__init__()

        self.clip_loss_lambda = clip_loss_lambda
        self.use_transport_plan_as_logits = use_transport_plan_as_logits

        self.clip_loss = ClipLoss()
        self.ot_loss = BatchLevelEntropicOTLoss(
            sinkhorn_solver=sinkhorn_solver,
            use_transport_plan_as_logits=self.use_transport_plan_as_logits,
            sim_matrix_temperature=sim_matrix_temperature,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_reg_m=sinkhorn_reg_m,
            sinkhorn_max_num_iters=sinkhorn_max_num_iters,
            sigrot_unbalanced_variant=sigrot_unbalanced_variant,
        )

    @property
    def last_transport_stats(self) -> dict[str, float | str]:
        return self.ot_loss.last_transport_stats

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        image_ids: Tensor | None = None,
        sim_matrix: Tensor | None = None,
        output_dict: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        clip_loss_value = self.clip_loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            image_ids=image_ids,
            output_dict=False,
            reduction=reduction,
        )

        ot_loss_value = self.ot_loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            output_dict=False,
            sim_matrix=sim_matrix,
            image_ids=image_ids,
            reduction=reduction,
        )
        total_loss = self.clip_loss_lambda * clip_loss_value + ot_loss_value

        return {"loss": total_loss} if output_dict else total_loss


class HybridSigLipTPLoss(nn.Module):
    """
    Combines SigLIP loss with Batch-level Entropic Optimal Transport Loss.
    """

    def __init__(
        self,
        sig_lip_loss_lambda: float = 0.1,
        sinkhorn_solver: Literal["sinkhorn", "sinkhorn_unbalanced"] = "sinkhorn",
        use_transport_plan_as_logits: bool = False,
        sim_matrix_temperature: float = 0.05,
        sinkhorn_reg: float = 0.05,
        sinkhorn_reg_m: float = 0.5,
        sinkhorn_max_num_iters: int = 200,
        sigrot_unbalanced_variant: Literal[
            "raw_gkl",
            "row_norm_mass_weighted",
            "mass_matched_gkl",
        ] = "raw_gkl",
    ):
        """
        In case `sim_matrix` is not provided in the forward pass: If `use_transport_plan_as_logits` is True,
        the transport plan will be used as logits for computing the cross-entropy loss,
        otherwise, use transport plan as soft labels.

        If `sim_matrix` is provided in the forward pass, `use_transport_plan_as_logits` takes no effect.
        """

        super().__init__()

        self.sig_lip_loss_lambda = sig_lip_loss_lambda
        self.use_transport_plan_as_logits = use_transport_plan_as_logits

        self.sig_lip_loss = SigLipLoss()
        self.ot_loss = BatchLevelEntropicOTLoss(
            sinkhorn_solver=sinkhorn_solver,
            use_transport_plan_as_logits=self.use_transport_plan_as_logits,
            sim_matrix_temperature=sim_matrix_temperature,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_reg_m=sinkhorn_reg_m,
            sinkhorn_max_num_iters=sinkhorn_max_num_iters,
            sigrot_unbalanced_variant=sigrot_unbalanced_variant,
        )

    @property
    def last_transport_stats(self) -> dict[str, float | str]:
        return self.ot_loss.last_transport_stats

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        image_ids: Tensor | None = None,
        sim_matrix: Tensor | None = None,
        output_dict: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        sig_lip_loss_value = self.sig_lip_loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            image_ids=image_ids,
            output_dict=False,
            reduction=reduction,
        )

        ot_loss_value = self.ot_loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            output_dict=False,
            sim_matrix=sim_matrix,
            image_ids=image_ids,
            reduction=reduction,
        )
        total_loss = self.sig_lip_loss_lambda * sig_lip_loss_value + ot_loss_value

        return {"loss": total_loss} if output_dict else total_loss
