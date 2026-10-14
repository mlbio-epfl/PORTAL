"""Baseline projector initializations.

Only the Random baseline from the main paper (Table 1) is included: a standard
Kaiming-uniform initialization of the 2-layer MLP, representing the "skip
projector alignment entirely" condition.
"""

import math

import torch
from loguru import logger
from torch import Tensor

from .construct import make_projector_state_dict

__all__ = ["random_baseline"]


def random_baseline(vision_dim: int, llm_dim: int, seed: int = 42) -> dict[str, Tensor]:
    """Kaiming-uniform random projector (PyTorch nn.Linear default)."""
    logger.info("building random baseline projector (seed={})", seed)
    gen = torch.Generator().manual_seed(seed)

    def _linear(out_dim: int, in_dim: int) -> tuple[Tensor, Tensor]:
        bound = 1.0 / math.sqrt(in_dim)
        w = torch.empty(out_dim, in_dim).uniform_(-bound, bound, generator=gen)
        b = torch.empty(out_dim).uniform_(-bound, bound, generator=gen)
        return w, b

    w1, b1 = _linear(llm_dim, vision_dim)
    w2, b2 = _linear(llm_dim, llm_dim)
    return make_projector_state_dict(w1, b1, w2, b2)
