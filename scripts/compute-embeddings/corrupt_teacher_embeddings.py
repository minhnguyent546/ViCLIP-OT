#!/usr/bin/env python

# pyright: reportPrivateImportUsage=false

"""Build corrupted or random teacher embeddings for SIGROT graph-quality ablations.

Reads the same ``(N, d)`` ``.pt`` tensors that ``viclip_ot/train.py`` loads via
``--precomputed_{image,caption}_embeddings_path`` and writes new tensors in that
format. Two modes:

* ``noise``: isotropic Gaussian noise on unit rows, then re-normalize.
* ``random``: i.i.d. Gaussian rows, then re-normalize (structureless control).

Formulas
--------

Rows are treated as unit vectors ``e \\in R^d``, ``||e||_2 = 1``.
For Qwen3-VL-Embedding-2B, ``d = 2048``.

**Noise.** Draw ``\\epsilon ~ N(0, I_d)`` and set

    tilde{e} = (e + \\sigma \\epsilon) / ||e + \\sigma \\epsilon||_2.                 (1)

``\\sigma`` is fixed from a target mean self-similarity
``\\rho* := E[<e, tilde{e}>] \\in (0, 1]``, not chosen as a free scale.

With ``\\epsilon ~ N(0, I_d)``, one has ``<e, \\epsilon> ~ N(0, 1)`` and
``||\\epsilon||_2^2 ~ \\chi^2_d``. The self-cosine expands as

    <e, tilde{e}>
        = (1 + \\sigma <e,\\epsilon>)
          / sqrt(1 + 2\\sigma <e,\\epsilon> + \\sigma^2 ||\\epsilon||_2^2).          (2)

For large ``d``, ``||\\epsilon||_2^2 / d \\to 1`` in probability and
``<e,\\epsilon> = O_p(1)`` is negligible next to ``\\sigma^2 d``, so

    E[<e, tilde{e}>]  \\approx  1 / sqrt(1 + \\sigma^2 d).                        (3)

Inverting (3) at a chosen ``\\rho*`` gives

    \\sigma(\\rho*, d)  =  sqrt( (1/(\\rho*)^2 - 1) / d ).                           (4)

Default: ``\\rho* = 0.5``, ``d = 2048`` \\Rightarrow ``\\sigma = sqrt(3/2048) \\approx 0.038273``.
At this ``d``, Monte Carlo self-cosine under (1)+(4) matches ``\\rho*`` within
``10^{-3}``. The script always logs the empirical mean self-cosine on the
loaded tensors.

For two clean unit vectors with cosine ``s = <u, v>``, independent corruptions
give the large-``d`` pairwise shrinkage

    E[<tilde{u}, tilde{v}>]  \\approx  s / (1 + \\sigma^2 d)  =  s (\\rho*)^2.     (5)

At ``\\rho* = 0.5`` Gram entries shrink by about ``1/4`` (e.g. ``s = 0.8`` becomes
``\\approx 0.2``). Neighborhood structure is weakened, not erased; ``random`` is
the structureless extreme (``\\sigma \\to \\infty`` limit of (1)).

Isotropic Gaussian is the max-entropy zero-mean law at fixed noise power
``E[||\\sigma \\epsilon||_2^2] = \\sigma^2 d`` once no preferred axis is allowed
(``\\Sigma = \\sigma^2 I``). It is rotation-invariant
(``R\\epsilon =^d \\epsilon`` for ``R \\in O(d)``), so it does not invent a false
semantic direction. Corrupting embeddings and rebuilding ``G_cross`` as in
training keeps ``G_cross`` a Gram combination; entrywise noise on ``G_cross``
would not.

**Random.** Ignore input values (shape only) and draw

    tilde{e}_i  =  g_i / ||g_i||_2,   g_i ~ N(0, I_d).                          (6)

Examples
--------

::

    python scripts/compute-embeddings/corrupt_teacher_embeddings.py \\
        --mode noise --rho_star 0.5 --seed 42 \\
        --image_embeddings  path/to/train_image_embeddings.pt \\
        --caption_embeddings path/to/train_caption_embeddings.pt

    python scripts/compute-embeddings/corrupt_teacher_embeddings.py \\
        --mode random --seed 42 \\
        --image_embeddings  path/to/train_image_embeddings.pt \\
        --caption_embeddings path/to/train_caption_embeddings.pt

    # stats only
    python scripts/compute-embeddings/corrupt_teacher_embeddings.py \\
        --mode noise --rho_star 0.5 --dry_run \\
        --image_embeddings ... --caption_embeddings ...
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from loguru import logger


def sigma_from_rho_star(rho_star: float, dim: int) -> float:
    """``\\sigma`` from target self-cosine via Eq. (4).

    Uses ``E[<e, tilde e>] \\approx 1/sqrt(1 + \\sigma^2 d)``, so
    ``\\sigma = sqrt((1/\\rho*^2 - 1) / d)`` for ``\\rho* \\in (0, 1]``.
    """
    if not (0.0 < rho_star <= 1.0):
        raise ValueError(f"rho_star must be in (0, 1], got {rho_star}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if rho_star == 1.0:
        return 0.0
    return math.sqrt((1.0 / (rho_star * rho_star) - 1.0) / float(dim))


def approx_expected_self_cosine(sigma: float, dim: int) -> float:
    """Large-``d`` approximation of ``E[<e, tilde e>]``, Eq. (3)."""
    return 1.0 / math.sqrt(1.0 + (sigma * sigma) * float(dim))


def approx_pairwise_shrinkage(rho_star: float) -> float:
    """Factor on clean pairwise cosines under noise, Eq. (5): ``(\\rho*)^2``."""
    return rho_star * rho_star


def _load_embedding_tensor(path: str) -> torch.Tensor:
    """Load a 2-D float tensor from a ``.pt`` file.

    Accepts a raw tensor (training format) or a dict holding one embedding
    tensor under a common key.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        tensor = obj
    elif isinstance(obj, dict):
        for key in (
            "embeddings",
            "embedding",
            "image_embeddings",
            "caption_embeddings",
            "text_embeddings",
            "features",
        ):
            if key in obj and isinstance(obj[key], torch.Tensor):
                tensor = obj[key]
                break
        else:
            tensor_items = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if len(tensor_items) == 1:
                tensor = next(iter(tensor_items.values()))
            else:
                raise TypeError(
                    f"{path}: expected a Tensor or a dict with one embedding "
                    f"tensor, got dict keys={list(obj.keys())}"
                )
    else:
        raise TypeError(f"{path}: expected Tensor or dict, got {type(obj)}")

    if tensor.ndim != 2:
        raise ValueError(
            f"{path}: expected 2-D embeddings (N, d), got shape {tuple(tensor.shape)}"
        )
    return tensor.detach().float().contiguous()


def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalize (cosine geometry for SIGROT graphs)."""
    return F.normalize(x, p=2, dim=-1, eps=eps)


def corrupt_isotropic_gaussian(
    embeddings: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Eq. (1): ``tilde e = (e + \\sigma eps) / ||e + \\sigma eps||_2``.

    Rows of ``embeddings`` must already be unit vectors.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0.0:
        return embeddings.clone()

    eps = torch.randn(
        embeddings.shape,
        generator=generator,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    return l2_normalize_rows(embeddings + sigma * eps)


def sample_random_unit_embeddings(
    num_rows: int,
    dim: int,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Eq. (6): i.i.d. Gaussian rows, L2-normalized."""
    if device is None:
        device = torch.device("cpu")
    g = torch.randn(num_rows, dim, generator=generator, dtype=dtype, device=device)
    return l2_normalize_rows(g)


def mean_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row cosine ``<a_i, b_i>`` (rows assumed unit-normalized)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a * b).sum(dim=-1).mean().item())


def mean_row_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x, ord=2, dim=-1).mean().item())


def default_output_path(input_path: str, mode: str, rho_star: float | None) -> str:
    """Output path next to the input: ``<stem>_noise_rho{ρ*}.pt`` or ``_random.pt``."""
    if mode == "noise":
        if rho_star is None:
            raise ValueError("rho_star is required for noise mode default paths")
        tag = f"noise_rho{rho_star:g}"
    elif mode == "random":
        tag = "random"
    else:
        raise ValueError(mode)
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_{tag}.pt"


def save_tensor(path: str, tensor: torch.Tensor, overwrite: bool) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (pass --overwrite)")
    torch.save(tensor.cpu(), path)


def save_sidecar_json(path: str, payload: dict, overwrite: bool) -> None:
    """Write JSON metadata next to a ``.pt`` output."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (pass --overwrite)")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def corrupt_embeddings(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.image_embeddings):
        logger.error(f"image embeddings not found: {args.image_embeddings}")
        return 1
    if not os.path.isfile(args.caption_embeddings):
        logger.error(f"caption embeddings not found: {args.caption_embeddings}")
        return 1

    image = _load_embedding_tensor(args.image_embeddings)
    caption = _load_embedding_tensor(args.caption_embeddings)

    if image.shape != caption.shape:
        logger.error(
            f"shape mismatch: image {tuple(image.shape)} vs caption {tuple(caption.shape)}"
        )
        return 1

    n_rows, dim = image.shape
    if args.expect_dim > 0 and dim != args.expect_dim:
        logger.error(
            f"expected d={args.expect_dim}, got d={dim}. "
            f"Pass --expect_dim {dim} (or 0) if intentional."
        )
        return 1

    image_unit = l2_normalize_rows(image)
    caption_unit = l2_normalize_rows(caption)
    clean_image_mean_norm = mean_row_norm(image)
    clean_caption_mean_norm = mean_row_norm(caption)

    gen_image = torch.Generator(device="cpu").manual_seed(args.seed)
    # Separate stream so image and caption noise are not identical.
    gen_caption = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    rho_star_nominal: float | None
    sigma: float | None
    approx_rho: float | None
    shrinkage: float | None

    if args.mode == "noise":
        if args.sigma is not None:
            if args.sigma < 0.0:
                logger.error(f"--sigma must be >= 0, got {args.sigma}")
                return 1
            sigma = float(args.sigma)
            rho_star_nominal = None
            approx_rho = approx_expected_self_cosine(sigma, dim)
        else:
            rho_star_nominal = float(args.rho_star)
            try:
                sigma = sigma_from_rho_star(rho_star_nominal, dim)
            except ValueError as exc:
                logger.error(str(exc))
                return 1
            approx_rho = approx_expected_self_cosine(sigma, dim)

        shrinkage = approx_rho * approx_rho
        image_out = corrupt_isotropic_gaussian(image_unit, sigma, gen_image)
        caption_out = corrupt_isotropic_gaussian(caption_unit, sigma, gen_caption)
        emp_rho_image = mean_row_cosine(image_unit, image_out)
        emp_rho_caption = mean_row_cosine(caption_unit, caption_out)
    else:
        sigma = None
        rho_star_nominal = None
        approx_rho = 0.0
        shrinkage = 0.0
        image_out = sample_random_unit_embeddings(n_rows, dim, gen_image, dtype=image_unit.dtype)
        caption_out = sample_random_unit_embeddings(
            n_rows, dim, gen_caption, dtype=caption_unit.dtype
        )
        emp_rho_image = mean_row_cosine(image_unit, image_out)
        emp_rho_caption = mean_row_cosine(caption_unit, caption_out)

    out_image = args.output_image or default_output_path(
        args.image_embeddings,
        args.mode,
        args.rho_star if args.mode == "noise" else None,
    )
    out_caption = args.output_caption or default_output_path(
        args.caption_embeddings,
        args.mode,
        args.rho_star if args.mode == "noise" else None,
    )

    logger.info("corrupt_teacher_embeddings")
    logger.info(f"mode                 : {args.mode}")
    logger.info(f"seed                 : {args.seed}  (caption uses seed+1)")
    logger.info(f"input image          : {args.image_embeddings}")
    logger.info(f"input caption        : {args.caption_embeddings}")
    logger.info(f"shape (N, d)         : ({n_rows}, {dim})")
    logger.info(
        f"clean mean ||row||   : image={clean_image_mean_norm:.6f}  "
        f"caption={clean_caption_mean_norm:.6f}"
    )
    if args.mode == "noise":
        rho_disp = f"{rho_star_nominal}" if rho_star_nominal is not None else "(sigma override)"
        logger.info(f"rho_star (nominal)   : {rho_disp}")
        logger.info(f"sigma                : {sigma:.8f}")
        logger.info(f"approx E[<e,tilde e>]: {approx_rho:.6f}   # Eq. (3)")
        logger.info(f"approx pairwise factor: {shrinkage:.6f}   # Eq. (5): (rho)^2")
        logger.info("sigma formula        : sqrt((1/rho*^2 - 1)/d)   # Eq. (4)")
    logger.info(f"empirical rho image  : {emp_rho_image:.6f}")
    logger.info(f"empirical rho caption: {emp_rho_caption:.6f}")
    logger.info(f"output image         : {out_image}")
    logger.info(f"output caption       : {out_caption}")

    if args.mode == "noise" and approx_rho is not None:
        for name, emp in (("image", emp_rho_image), ("caption", emp_rho_caption)):
            if abs(emp - approx_rho) > 0.05:
                logger.warning(
                    f"empirical rho on {name} ({emp:.4f}) differs from "
                    f"approx target ({approx_rho:.4f}) by > 0.05. "
                    f"Check that inputs are unit-norm rows."
                )

    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "seed_caption": args.seed + 1,
        "num_rows": n_rows,
        "dim": dim,
        "rho_star_nominal": rho_star_nominal,
        "sigma": sigma,
        "approx_expected_self_cosine": approx_rho,
        "approx_pairwise_shrinkage_factor": shrinkage,
        "empirical_rho_image": emp_rho_image,
        "empirical_rho_caption": emp_rho_caption,
        "input_image_embeddings": os.path.abspath(args.image_embeddings),
        "input_caption_embeddings": os.path.abspath(args.caption_embeddings),
        "output_image_embeddings": os.path.abspath(out_image),
        "output_caption_embeddings": os.path.abspath(out_caption),
        "formulas": {
            "corruption": ("tilde_e = (e + sigma * eps) / ||e + sigma * eps||_2, eps ~ N(0, I_d)"),
            "sigma_from_rho_star": "sigma = sqrt( (1/rho_star^2 - 1) / d )",
            "expected_self_cosine_large_d": ("E[<e, tilde_e>] ~= 1 / sqrt(1 + sigma^2 * d)"),
            "pairwise_shrinkage_large_d": ("E[<tilde_u, tilde_v>] ~= <u,v> * (rho)^2"),
            "random": "tilde_e = g / ||g||_2, g ~ N(0, I_d)",
        },
    }

    if args.dry_run:
        logger.info("dry-run: no files written")
        logger.info(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    try:
        save_tensor(out_image, image_out, overwrite=args.overwrite)
        save_tensor(out_caption, caption_out, overwrite=args.overwrite)
        if not args.no_sidecar:
            out_image_json = os.path.splitext(out_image)[0] + ".json"
            out_caption_json = os.path.splitext(out_caption)[0] + ".json"
            save_sidecar_json(out_image_json, metadata, overwrite=args.overwrite)
            save_sidecar_json(out_caption_json, metadata, overwrite=args.overwrite)
    except FileExistsError as exc:
        logger.error(str(exc))
        return 1

    logger.info("wrote:")
    logger.info(f"  {out_image}  shape={tuple(image_out.shape)}")
    logger.info(f"  {out_caption}  shape={tuple(caption_out.shape)}")
    if not args.no_sidecar:
        logger.info(f"  {os.path.splitext(out_image)[0] + '.json'}")
        logger.info(f"  {os.path.splitext(out_caption)[0] + '.json'}")
    return 0


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("noise", "random"),
        required=True,
        help=(
            "noise: isotropic Gaussian on unit rows, then re-normalize (Eq. 1). "
            "random: i.i.d. Gaussian rows, then re-normalize (Eq. 6)."
        ),
    )
    parser.add_argument(
        "--image_embeddings",
        type=str,
        required=True,
        help="Input image teacher embeddings .pt of shape (N, d).",
    )
    parser.add_argument(
        "--caption_embeddings",
        type=str,
        required=True,
        help="Input caption teacher embeddings .pt of shape (N, d).",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        help="Output image .pt. Default: <input>_noise_rho{ρ*}.pt or _random.pt",
        default=None,
    )
    parser.add_argument(
        "--output_caption",
        type=str,
        help="Output caption .pt. Default: <input>_noise_rho{ρ*}.pt or _random.pt",
        default=None,
    )
    parser.add_argument(
        "--rho_star",
        type=float,
        help=(
            "Target E[<e, tilde e>] for --mode noise. "
            "Sets \\sigma via Eq. (4). Ignored for --mode random."
        ),
        default=0.5,
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help=(
            "Override \\sigma for --mode noise. Prefer --rho_star so severity "
            "does not depend on d. Empirical \\rho is still reported."
        ),
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed (caption stream uses seed+1).",
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output .pt / .json files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Log statistics only; write nothing.",
    )
    parser.add_argument(
        "--no_sidecar",
        action="store_true",
        help="Skip writing companion .json metadata.",
    )
    parser.add_argument(
        "--expect_dim",
        type=int,
        help=(
            "Require embedding dim d to match this value "
            "(Qwen3-VL-Embedding-2B is 2048). Pass 0 to skip."
        ),
        default=2048,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Corrupt or replace precomputed teacher embeddings for SIGROT "
            "graph-quality ablations. Formulas are in the module docstring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_opts(parser)
    args = parser.parse_args()

    return corrupt_embeddings(args)


if __name__ == "__main__":
    raise SystemExit(main())
