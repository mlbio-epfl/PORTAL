"""The PORTAL closed-form projector construction (Eq. 2).

Builds the LLaVA projector as the Gaussian optimal-transport map between the
vision and LLM feature distributions on their shared top-k principal subspace.
The map is placed in the first linear layer; the second is fixed to the
identity. See build_portal.
"""

import torch
from loguru import logger
from torch import Tensor

from .stats import Stats

__all__ = ["build_portal", "make_projector_state_dict", "PROJECTOR_KEYS"]

# LLaVA's projector is nn.Sequential(Linear, GELU, Linear). These are the keys a
# LLaVA checkpoint expects, in order (W1, b1, W2, b2); the single source of truth
# shared with portal.io.
PROJECTOR_KEYS = (
    "model.mm_projector_f.0.weight",
    "model.mm_projector_f.0.bias",
    "model.mm_projector_f.2.weight",
    "model.mm_projector_f.2.bias",
)


def make_projector_state_dict(
    w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor
) -> dict[str, Tensor]:
    """Pack the two linear layers into a LLaVA projector state_dict (bfloat16)."""
    return {
        key: t.to(torch.bfloat16)
        for key, t in zip(PROJECTOR_KEYS, (w1, b1, w2, b2))
    }


def build_portal(
    source: Stats,
    target: Stats,
    max_scale_ratio: float | None = 100.0,
) -> dict[str, Tensor]:
    """Build a PORTAL projector state_dict (Eq. 2) from source/target stats.

    `source` and `target` are (mu, V, sigma) triples from pca_stats for the
    vision and language sides. `max_scale_ratio` clamps each per-axis scale
    sigma_y/sigma_x to curb blow-up from near-degenerate source directions (the
    paper uses 100.0; None disables the clamp).
    """
    mu_x, v_x, sigma_x = source
    mu_y, v_y, sigma_y = target

    k = min(v_x.shape[0], v_y.shape[0], sigma_x.shape[0], sigma_y.shape[0])
    v_x, sigma_x = v_x[:k], sigma_x[:k]
    v_y, sigma_y = v_y[:k], sigma_y[:k]

    # 1e-8 guards against zero-variance (degenerate) source directions.
    scale = sigma_y / (sigma_x + 1e-8)
    if max_scale_ratio is not None:
        n_clamped = int((scale > max_scale_ratio).sum().item())
        scale = scale.clamp(max=max_scale_ratio)
        if n_clamped:
            logger.info(
                "clamped {}/{} scale ratios to max={}", n_clamped, k, max_scale_ratio
            )

    w1 = v_y.T @ torch.diag(scale) @ v_x
    b1 = mu_y - w1 @ mu_x

    d_t = v_y.shape[1]
    w2 = torch.eye(d_t)
    b2 = torch.zeros(d_t)

    logger.info(
        "PORTAL map built: k={}, d_src={}, d_tgt={}, scale range=[{:.4f}, {:.4f}]",
        k,
        v_x.shape[1],
        d_t,
        scale.min().item(),
        scale.max().item(),
    )
    return make_projector_state_dict(w1, b1, w2, b2)
