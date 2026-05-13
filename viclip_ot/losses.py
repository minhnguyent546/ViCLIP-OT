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
    metric_cost_matrix,
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


def _generalized_kl(P: Tensor, Q: Tensor, eps: float | None = None) -> Tensor:
    """
    Compute generalized KL divergence between true probability distribution ``P`` and
    approximated probability distribution ``Q``.
    """

    if eps is None:
        eps = torch.finfo(P.dtype).eps

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
        self.sigrot_unbalanced_variant = sigrot_unbalanced_variant
        self.last_transport_stats: dict[str, float | str] = {}

        logger.info(f"Using Sinkhorn Solver: {self.sinkhorn_solver}")
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
                    reg=0.05,
                    max_num_iters=200,
                )
            elif self.sinkhorn_solver == "sinkhorn_unbalanced":
                transport_plan_i2t = _sinkhorn_unbalanced(
                    metric_cost_matrix=cost_matrix_i2t,
                    reg=0.05,
                    reg_m=0.5,  # cosine similarity scores < (1 - 0.5) = 0.5 will be considered as dissimilar/noisy
                    max_num_iters=200,
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
                        "i2t_mass_mean": i2t_mass.mean(),
                        "i2t_mass_std": i2t_mass.std(unbiased=False),
                        "i2t_mass_min": i2t_mass.min(),
                        "i2t_mass_max": i2t_mass.max(),
                        "t2i_mass_mean": t2i_mass.mean(),
                        "t2i_mass_std": t2i_mass.std(unbiased=False),
                        "t2i_mass_min": t2i_mass.min(),
                        "t2i_mass_max": t2i_mass.max(),
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
                prob_i2t = transport_plan_i2t
                prob_t2i = transport_plan_t2i
                loss_i2t = _generalized_kl(prob_i2t, sim_i2t)
                loss_t2i = _generalized_kl(prob_t2i, sim_t2i)
                if reduction == "mean":
                    loss_i2t = loss_i2t.sum() / batch_size
                    loss_t2i = loss_t2i.sum() / batch_size
                elif reduction == "sum":
                    loss_i2t = loss_i2t.sum()
                    loss_t2i = loss_t2i.sum()
                elif reduction == "none":
                    pass
                else:
                    raise ValueError(
                        f"Unsupported reduction: {reduction}. Expected one of ['mean', 'sum', 'none']."
                    )
            elif self.sigrot_unbalanced_variant == "row_norm_mass_weighted":
                # `row_norm_mass_weighted`: normalize the unbalanced plan into proper row-wise
                # distributions, then use detached row/column mass as confidence.
                # This prevents KL from comparing incompatible mass scales while
                # still letting unbalanced OT downweight noisy/ambiguous rows.
                plan_i2t = transport_plan_i2t / i2t_mass.clamp_min(eps)
                plan_t2i = transport_plan_t2i / t2i_mass.clamp_min(eps)

                kl_i2t = _generalized_kl(plan_i2t, sim_i2t).sum(dim=1)
                kl_t2i = _generalized_kl(plan_t2i, sim_t2i).sum(dim=1)

                weights_i2t = i2t_mass.squeeze(1).detach()
                weights_t2i = t2i_mass.squeeze(1).detach()
                if reduction == "mean":
                    loss_i2t = (weights_i2t * kl_i2t).sum() / weights_i2t.sum().clamp_min(eps)
                    loss_t2i = (weights_t2i * kl_t2i).sum() / weights_t2i.sum().clamp_min(eps)
                elif reduction == "sum":
                    loss_i2t = (weights_i2t * kl_i2t).sum()
                    loss_t2i = (weights_t2i * kl_t2i).sum()
                elif reduction == "none":
                    loss_i2t = weights_i2t * kl_i2t
                    loss_t2i = weights_t2i * kl_t2i
                else:
                    raise ValueError(
                        f"Unsupported reduction: {reduction}. Expected one of ['mean', 'sum', 'none']."
                    )
            elif self.sigrot_unbalanced_variant == "mass_matched_gkl":
                # `mass_matched_gkl`: keep the unbalanced plan as a measure, but scale the
                # teacher graph by detached OT mass so P and Q have compatible row
                # totals. This is measure-level GKL while avoiding the raw baseline's
                # accidental penalty against a fixed row-sum-1 teacher distribution.
                sim_measure_i2t = i2t_mass.detach() * sim_i2t
                sim_measure_t2i = t2i_mass.detach() * sim_t2i
                prob_i2t = transport_plan_i2t
                prob_t2i = transport_plan_t2i

                loss_i2t = _generalized_kl(prob_i2t, sim_measure_i2t)
                loss_t2i = _generalized_kl(prob_t2i, sim_measure_t2i)

                if reduction == "mean":
                    loss_i2t = loss_i2t.sum() / i2t_mass.detach().sum().clamp_min(eps)
                    loss_t2i = loss_t2i.sum() / t2i_mass.detach().sum().clamp_min(eps)
                elif reduction == "sum":
                    loss_i2t = loss_i2t.sum()
                    loss_t2i = loss_t2i.sum()
                elif reduction == "none":
                    pass
                else:
                    raise ValueError(
                        f"Unsupported reduction: {reduction}. Expected one of ['mean', 'sum', 'none']."
                    )
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
