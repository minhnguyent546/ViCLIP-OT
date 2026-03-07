import contextlib
from typing import Literal

import ot
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch import Tensor


class ClipLoss(nn.Module):
    """(Open) CLIP loss.

    Taken and modified from: https://github.com/mlfoundations/open_clip/blob/d3cdb734a2710feeb4c6307df037afa5f786a3e1/src/open_clip/loss.py#L68.

    Modification:
    - Add type hints.
    - Removed multi-node support for simplicity.
    """

    def __init__(self):
        super().__init__()

    def get_logits(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
    ):
        # Compute matrix multiplication in float32 for numerical stability
        # with mixed precision training, then cast back to original dtype
        original_dtype = image_features.dtype
        logits_per_image = logit_scale * (image_features.float() @ text_features.float().T)
        logits_per_text = logits_per_image.T

        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias
            logits_per_text = logits_per_text + logit_bias

        return logits_per_image.to(original_dtype), logits_per_text.to(original_dtype)

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
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
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

        print(f"Using Sinkhorn Solver: {self.sinkhorn_solver}")

    def get_logits(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
    ):
        # Compute matrix multiplication in float32 for numerical stability
        # with mixed precision training, then cast back to original dtype
        original_dtype = image_features.dtype
        logits_per_image = logit_scale * (image_features.float() @ text_features.float().T)
        logits_per_text = logits_per_image.T

        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias
            logits_per_text = logits_per_text + logit_bias

        return logits_per_image.to(original_dtype), logits_per_text.to(original_dtype)

    def sinkhorn(
        self,
        metric_cost_matrix: Tensor,
        reg: float = 0.05,  # try with [0.01, 0.1] with normalized cost matrix
        max_num_iters: int = 200,
    ) -> Tensor:
        """Solve the unbalanced entropic regularization optimal transport problem and return the OT plan."""
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

    def sinkhorn_unbalanced(
        self,
        metric_cost_matrix,
        reg: float = 0.05,  # try with [0.01, 0.1] with normalized cost matrix
        reg_m: float = 0.5,  # try with [0.3, 0.8] with normalized cost matrix
        max_num_iters: int = 200,
    ):
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
            # compute raw cosine similarity (without scaling and bias)
            raw_cosine_sim = image_features @ text_features.T
            cost_matrix = 1 - raw_cosine_sim  # values are in range [0, 2]
            if self.sinkhorn_solver == "sinkhorn":
                transport_plan = self.sinkhorn(
                    metric_cost_matrix=cost_matrix,
                    reg=0.05,
                    max_num_iters=200,
                )
            elif self.sinkhorn_solver == "sinkhorn_unbalanced":
                transport_plan = self.sinkhorn_unbalanced(
                    metric_cost_matrix=cost_matrix,
                    reg=0.05,
                    reg_m=0.5,  # cosine similarity scores < (1 - 0.5) = 0.5 will be considered as dissimilar/noisy
                    max_num_iters=200,
                )
            else:
                raise ValueError(f"Unsupported solver: {self.sinkhorn_solver}")

        if sim_matrix is not None:
            #  Generalized KL Divergence (transport_plan || sim_matrix)
            # NOTE: note that this is reverse KL divergence
            # TODO: find a better way to scale sim_matrix for stable training other than using `logit_scale`
            sim_matrix = (logit_scale * sim_matrix).softmax(dim=1)
            loss_i2t = transport_plan * (transport_plan.log() - sim_matrix.log() - 1) + sim_matrix
            loss_t2i = (
                transport_plan.t() * (transport_plan.t().log() - sim_matrix.t().log() - 1)
                + sim_matrix.t()
            )
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
        else:
            device = image_features.device
            if self.use_transport_plan_as_logits:
                # use transport plan as logits
                if image_ids is None:
                    # 1-1 matching between image and text
                    labels = torch.arange(batch_size, device=device, dtype=torch.int64)

                    loss_i2t = Fun.cross_entropy(
                        input=transport_plan, target=labels, reduction=reduction
                    )
                    loss_t2i = Fun.cross_entropy(
                        input=transport_plan.T, target=labels, reduction=reduction
                    )
                else:
                    image_ids = image_ids.to(device)
                    # shape: (batch_size, batch_size)
                    matches = image_ids.view(-1, 1) == image_ids.view(1, -1)

                    # create soft labels (probs)
                    soft_labels = matches.float()
                    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
                    loss_i2t = Fun.cross_entropy(transport_plan, soft_labels, reduction=reduction)
                    loss_t2i = Fun.cross_entropy(
                        transport_plan.T, soft_labels, reduction=reduction
                    )
            else:
                # use transport plan as soft targets

                logits_per_image, logits_per_text = self.get_logits(
                    image_features,
                    text_features,
                    logit_scale,
                    logit_bias=logit_bias,
                )

                loss_i2t = Fun.cross_entropy(
                    input=logits_per_image,
                    target=transport_plan,
                    reduction=reduction,
                )
                loss_t2i = Fun.cross_entropy(
                    input=logits_per_text,
                    target=transport_plan.T,
                    reduction=reduction,
                )

        total_loss = (loss_i2t + loss_t2i) / 2

        return {"loss": total_loss} if output_dict else total_loss


class EmbeddingDistillationLoss(nn.Module):
    """
    Embedding Distillation Loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        projected_image_features: Tensor,
        projected_text_features: Tensor,
        teacher_image_features: Tensor,
        teacher_text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        output_dict: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        """
        Compute the embedding distillation loss as the sum of cosine distances
        between (projected) student and teacher embeddings for both modalities.
        """

        image_cosine_dist = 1 - Fun.cosine_similarity(
            projected_image_features.float(), teacher_image_features.float(), dim=-1
        )
        text_cosine_dist = 1 - Fun.cosine_similarity(
            projected_text_features.float(), teacher_text_features.float(), dim=-1
        )

        # sum over modalities {x, y} for each sample
        loss = 0.5 * (image_cosine_dist + text_cosine_dist)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported reduction: {reduction}. Expected one of ['mean', 'sum', 'none']."
            )

        return {"loss": loss} if output_dict else loss


class HybridClipTPLoss(nn.Module):
    """
    Combines CLIP loss with Batch-level Entropic Optimal Transport Loss.
    """

    def __init__(
        self,
        clip_loss_lambda: float = 0.1,
        sinkhorn_solver: Literal["sinkhorn", "sinkhorn_unbalanced"] = "sinkhorn",
        use_transport_plan_as_logits: bool = False,
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
        )

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
        )

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


class HybridClipTPDistillLoss(nn.Module):
    """
    Combines CLIP loss with Batch-level Entropic Optimal Transport Loss and embedding distillation loss.
    """

    def __init__(
        self,
        clip_loss_lambda: float = 0.1,
        ot_loss_lambda: float = 1.0,
        embedding_distillation_lambda: float = 2.0,
        sinkhorn_solver: Literal["sinkhorn", "sinkhorn_unbalanced"] = "sinkhorn",
        use_transport_plan_as_logits: bool = False,
    ):
        """
        In case `sim_matrix` is not provided in the forward pass: If `use_transport_plan_as_logits` is True,
        the transport plan will be used as logits for computing the cross-entropy loss,
        otherwise, use transport plan as soft labels.

        If `sim_matrix` is provided in the forward pass, `use_transport_plan_as_logits` takes no effect.
        """

        super().__init__()

        self.clip_loss_lambda = clip_loss_lambda
        self.ot_loss_lambda = ot_loss_lambda
        self.embedding_distillation_lambda = embedding_distillation_lambda
        self.use_transport_plan_as_logits = use_transport_plan_as_logits

        self.clip_loss = ClipLoss()
        self.ot_loss = BatchLevelEntropicOTLoss(
            sinkhorn_solver=sinkhorn_solver,
            use_transport_plan_as_logits=self.use_transport_plan_as_logits,
        )
        self.embedding_distillation_loss = EmbeddingDistillationLoss()

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        projected_image_features: Tensor,
        projected_text_features: Tensor,
        teacher_image_features: Tensor,
        teacher_text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        image_ids: Tensor | None = None,
        sim_matrix: Tensor | None = None,
        reduction: str = "mean",
        **kwargs,
    ):
        if self.clip_loss_lambda > 0:
            clip_loss_value = self.clip_loss(
                image_features=image_features,
                text_features=text_features,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                image_ids=image_ids,
                output_dict=False,
                reduction=reduction,
            )
        else:
            clip_loss_value = 0

        if self.ot_loss_lambda > 0:
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
        else:
            ot_loss_value = 0

        if self.embedding_distillation_lambda > 0:
            embedding_distillation_loss_value = self.embedding_distillation_loss(
                projected_image_features=projected_image_features,
                projected_text_features=projected_text_features,
                teacher_image_features=teacher_image_features,
                teacher_text_features=teacher_text_features,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                output_dict=False,
                reduction=reduction,
            )
        else:
            embedding_distillation_loss_value = 0

        return {
            "clip_loss": self.clip_loss_lambda * clip_loss_value,
            "ot_loss": self.ot_loss_lambda * ot_loss_value,
            "embedding_distillation_loss": self.embedding_distillation_lambda
            * embedding_distillation_loss_value,
        }
