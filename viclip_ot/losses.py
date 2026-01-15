from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch import Tensor


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss as in `Supervised Contrastive Learning` ([arXiv](https://arxiv.org/abs/2004.11362)).

    It also supports the unsupervised contrastive loss in [SimCLR](https://arxiv.org/abs/2002.05709).

    Taken and modified from: https://github.com/HobbitLong/SupContrast/blob/66a8fe53880d6a1084b2e4e0db0a019024d6d41a/losses.py#L11.

    Modification:
    - Add type hints.
    - Get `device` using `features.device`
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: Literal["all", "one"] = "all",
        base_temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: Tensor, labels: Tensor | None = None, mask: Tensor | None = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            assert mask is not None
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # prevent computing log(0), which will produce nan in the loss
        # see: https://github.com/HobbitLong/SupContrast/pull/111/changes/48140375921682b905915ec5c724bca5dd65c40e
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class XSampleContrastiveLoss(nn.Module):
    """X-Sample ContrastiveLoss as in `X-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs` ([arXiv](https://arxiv.org/abs/2407.18134)).

    Reference:
    """

    def __init__(self) -> None:
        super().__init__()

        raise NotImplementedError("XSampleContrastiveLoss is not yet implemented.")

    def forward(self): ...


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
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

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
        image_ids: Tensor | None = None,  # only for compatibility with ClipLoss
    ):
        loss = self._loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            reduction=reduction,
        )

        return {"loss": loss} if output_dict else loss


class EntropicOTLoss(nn.Module):
    """
    Drop-in replacement for ClipLoss that uses entropic optimal transport.
    This implementation is adapted from the OT-CLIP loss modules provided
    in https://github.com/fan23j/ICML2024-OT-CLIP (training/loss.py).:contentReference[oaicite:1]{index=1}

    Args:
        reg: Entropic regularization coefficient (larger → smoother plan).
        n_iters: Number of Sinkhorn iterations.
        eps: Small epsilon for numerical stability.

    Forward receives:
    - image_features, text_features
    - logit_scale, (optional) logit_bias
    - image_ids (for multi-caption soft labels)
    - output_dict flag to return dict with "loss"
    """

    def __init__(self, reg: float = 0.05, n_iters: int = 20, eps: float = 1e-8):
        super().__init__()
        self.reg = reg
        self.n_iters = n_iters
        self.eps = eps

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        """
        Compute CLIP-style logits by cosine-like similarity:
        - Use float32 matmul for better numerical stability
        - Scale by logit_scale
        - Optionally add logit_bias (if using adaptive bias)
        """
        original_dtype = image_features.dtype

        # compute similarity in float (stabilizes backward)
        logits_per_image = logit_scale * (
            image_features.float() @ text_features.float().T
        )
        logits_per_text = logits_per_image.T

        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias
            logits_per_text = logits_per_text + logit_bias

        # convert back to original dtype
        return (
            logits_per_image.to(original_dtype),
            logits_per_text.to(original_dtype),
        )

    def entropic_ot(self, cost, reg=None, n_iters=None):
        """
        Compute entropic optimal transport plan (Sinkhorn algorithm).

        Args:
            cost: (n, m) cost matrix (lower cost = better match).
        Returns:
            P: (n, m) transport plan approximately satisfying uniform marginals.
        """
        if reg is None:
            reg = self.reg
        if n_iters is None:
            n_iters = self.n_iters

        # Do Sinkhorn in float32 for stability (even if model is fp16/bf16)
        orig_dtype = cost.dtype
        cost = cost.float()

        device = cost.device
        n, m = cost.shape

        # Uniform marginals
        a = torch.full((n,), 1.0 / n, device=device, dtype=cost.dtype)
        b = torch.full((m,), 1.0 / m, device=device, dtype=cost.dtype)

        # Stabilize before exp:
        # shift by row-wise minimum
        cost = cost - cost.min(dim=1, keepdim=True).values

        # Gibbs kernel
        K = torch.exp(-cost / reg)

        u = torch.ones((n,), device=device, dtype=cost.dtype)
        v = torch.ones((m,), device=device, dtype=cost.dtype)

        for _ in range(n_iters):
            Kv = K @ v
            u = a / (Kv + self.eps)

            KTu = K.T @ u
            v = b / (KTu + self.eps)

        P = (u.unsqueeze(1) * K) * v.unsqueeze(0)

        # Optional: renormalize rows to sum to 1
        # P = P / (P.sum(dim=1, keepdim=True) + self.eps)

        return P.to(orig_dtype)


    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        image_ids=None,
        output_dict: bool = False,
        reduction: str = "mean",
    ):
        """
        Forward pass computing the OT-based loss.

        If image_ids is None:
            compute symmetric transport cross-entropy (like CLIP InfoNCE).
        Else:
            allow many-to-many soft label matching per the dataset IDs.
        """
        device = image_features.device

        # compute similarity logits
        logits_i2t, logits_t2i = self.get_logits(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

        # convert similarities to costs (lower cost = higher similarity)
        cost_i2t = -logits_i2t
        cost_t2i = -logits_t2i

        # compute transport plans
        P_i2t = self.entropic_ot(cost_i2t, reg=self.reg, n_iters=self.n_iters)
        P_t2i = self.entropic_ot(cost_t2i, reg=self.reg, n_iters=self.n_iters)

        # convert transport plans to logits via log (Softmax(log P) ≈ P)
        ot_logits_i2t = torch.log(P_i2t + self.eps)
        ot_logits_t2i = torch.log(P_t2i + self.eps)

        if image_ids is None:
            # standard symmetric cross entropy like CLIP
            labels = torch.arange(
                ot_logits_i2t.size(0), device=device, dtype=torch.long
            )
            loss_i2t = Fun.cross_entropy(ot_logits_i2t, labels, reduction=reduction)
            loss_t2i = Fun.cross_entropy(ot_logits_t2i, labels, reduction=reduction)
        else:
            # multi-caption soft labels: rows match same image_ids
            image_ids = image_ids.to(device)
            matches = image_ids.view(-1, 1) == image_ids.view(1, -1)
            soft_labels = matches.float()
            soft_labels = soft_labels / (
                soft_labels.sum(dim=1, keepdim=True) + self.eps
            )

            # compute cross entropy with soft targets
            logp_i2t = Fun.log_softmax(ot_logits_i2t, dim=1)
            logp_t2i = Fun.log_softmax(ot_logits_t2i, dim=1)

            loss_i2t = -(soft_labels * logp_i2t).sum(dim=1)
            loss_t2i = -(soft_labels * logp_t2i).sum(dim=1)

            if reduction == "mean":
                loss_i2t = loss_i2t.mean()
                loss_t2i = loss_t2i.mean()
            elif reduction == "sum":
                loss_i2t = loss_i2t.sum()
                loss_t2i = loss_t2i.sum()
            # else: "none" returns per-sample

        total_loss = (loss_i2t + loss_t2i) / 2.0
        return {"loss": total_loss} if output_dict else total_loss
