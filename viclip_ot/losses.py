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
        negative_only: bool = False,
        reduction: str = "mean",
    ) -> Tensor:
        # shape: (batch_size, batch_size)
        logits = self.get_logits(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )
        # shape: (batch_size, batch_size)
        labels = self.get_ground_truth(
            device=image_features.device,
            dtype=image_features.dtype,
            num_logits=image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -Fun.logsigmoid(labels * logits).sum()

        if reduction == "mean":
            loss = loss / image_features.shape[0]

        return loss

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        output_dict: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        loss = self._loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            reduction=reduction,
        )

        return {"loss": loss} if output_dict else loss


class BatchLevelEntropicOTLoss(nn.Module):
    """
    Batch-level Entropic Optimal Transport Loss
    """

    def __init__(
        self,
        sinkhorn_solver: Literal["sinkhorn", "sinkhorn_unbalanced", "fused_gromov"] = "sinkhorn",
    ):
        super().__init__()
        self.sinkhorn_solver = sinkhorn_solver
        self.fgw_alpha = 0.5

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
        reg_m: float = 0.4,  # try with [0.3, 0.8] with normalized cost matrix
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

        return transport_plan  # pyright: ignore[reportReturnType]

    def fused_gromov_solver(
        self,
        M: Tensor,
        C1: Tensor,
        C2: Tensor,
        reg: float = 0.01,
        max_num_iters: int = 200,
    ) -> Tensor:
        """
        Solve the Entropic Fused Gromov-Wasserstein problem.
        M: Inter-domain cost (Wasserstein)
        C1, C2: Intra-domain structure costs (Gromov)
        """
        device = M.device
        batch_size = M.shape[0]

        # Marginal distributions (Uniform)
        p = torch.ones((batch_size,), device=device) / batch_size
        q = torch.ones((batch_size,), device=device) / batch_size

        # Using square_loss as it fits feature distance better than kl_loss
        transport_plan = ot.gromov.entropic_fused_gromov_wasserstein(
            M=M,
            C1=C1,
            C2=C2,
            p=p,
            q=q,
            loss_fun="square_loss",
            epsilon=reg,
            alpha=self.fgw_alpha,
            numItermax=max_num_iters,
            verbose=False,
        )

        return transport_plan * batch_size  # pyright: ignore[reportReturnType]

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        logit_bias: Tensor | None = None,
        output_dict: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        """
        If `image_ids` is provided, the computation will take into account image with multiple captions.
        """
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        with torch.no_grad():
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
                    reg_m=0.4,  # cosine similarity scores < (1 - 0.4) = 0.6 will be considered as dissimilar/noisy
                    max_num_iters=200,
                )
            elif self.sinkhorn_solver == "fused_gromov":
                # Compute Intra-domain Structure Matrices (Gromov Term)
                # Use float32 to match precision of cost_matrix usually
                # Distance = 1 - Cosine Similarity
                C1 = 1 - (image_features @ image_features.T)
                C2 = 1 - (text_features @ text_features.T)

                transport_plan = self.fused_gromov_solver(
                    M=cost_matrix,
                    C1=C1,
                    C2=C2,
                    reg=0.01,  # FGW often prefers slightly smaller reg
                    max_num_iters=200,
                )
            else:
                raise ValueError(f"Unsupported solver: {self.sinkhorn_solver}")

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
        # # Use KL Divergence for soft labels
        # # KL(target || input) = sum(target * (log(target) - input))
        # # Here input is log_plan.
        # loss_i2t = Fun.kl_div(
        #     log_transport_plan,
        #     sim_matrix,
        #     reduction=("batchmean" if reduction == "mean" else reduction),
        #     log_target=False,
        # )
        # loss_t2i = Fun.kl_div(
        #     log_transport_plan.T,
        #     sim_matrix,
        #     reduction=("batchmean" if reduction == "mean" else reduction),
        #     log_target=False,
        # )

        #  Generalized KL Divergence (transport_plan || sim_matrix)
        # sim_matrix = (logit_scale * sim_matrix).softmax(dim=1) * batch_size
        # loss_i2t = transport_plan * (transport_plan.log() - sim_matrix.log() - 1) + sim_matrix
        # loss_t2i = (
        #     transport_plan.T * (transport_plan.T.log() - sim_matrix.T.log() - 1) + sim_matrix.T
        # )
        # if reduction == "mean":
        #     loss_i2t = loss_i2t.sum() / batch_size
        #     loss_t2i = loss_t2i.sum() / batch_size
        # elif reduction == "sum":
        #     loss_i2t = loss_i2t.sum()
        #     loss_t2i = loss_t2i.sum()
        # elif reduction == "none":
        #     pass
        # else:
        #     raise ValueError(
        #         f"Unsupported reduction: {reduction}. Expected one of ['mean', 'sum', 'none']."
        #     )

        total_loss = (loss_i2t + loss_t2i) / 2

        return {"loss": total_loss} if output_dict else total_loss


class HybridClipTPLoss(nn.Module):
    """
    Combines CLIP loss with Batch-level Entropic Optimal Transport Loss.
    """

    def __init__(
        self,
        ot_start_epoch: int,
        ot_loss_lambda: float = 1.0,
        sinkhorn_solver: Literal["sinkhorn", "sinkhorn_unbalanced"] = "sinkhorn",
    ):
        """
        Args:
            ot_start_epoch (int): Epoch to start applying the OT loss (0-based).
            ot_loss_lambda (float): Weighting factor for the OT loss component (`total_loss = clip_loss + lambda * ot_loss`).
        """
        super().__init__()

        self.ot_start_epoch = ot_start_epoch
        self.ot_loss_lambda = ot_loss_lambda

        self.clip_loss = ClipLoss()
        self.ot_loss = BatchLevelEntropicOTLoss(sinkhorn_solver=sinkhorn_solver)

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        logit_scale: Tensor,
        epoch: int,
        logit_bias: Tensor | None = None,
        image_ids: Tensor | None = None,
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

        if self.ot_start_epoch >= 0 and epoch >= self.ot_start_epoch:
            ot_loss_value = self.ot_loss(
                image_features=image_features,
                text_features=text_features,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                output_dict=False,
                reduction=reduction,
            )
            total_loss = clip_loss_value + self.ot_loss_lambda * ot_loss_value
        else:
            total_loss = clip_loss_value

        return {"loss": total_loss} if output_dict else total_loss
