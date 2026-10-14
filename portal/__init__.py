"""PORTAL: a training-free projector for multimodal LLMs via optimal transport.

Reference implementation for "Rethinking Projector Training in Multimodal LLMs"
(NeurIPS 2026). PORTAL initializes the LLaVA projector in closed form from the
per-modality feature statistics of a vision encoder and an LLM, using neither
paired image-caption data nor gradient-based optimization.

    from portal import construct_portal, save_projector
    sd = construct_portal(image_dir="coco/val2017", llm_model="llava-hf/llava-1.5-7b-hf")
    save_projector(sd, "checkpoints/portal_vicuna_clip")
"""

from .baselines import random_baseline
from .construct import build_portal, make_projector_state_dict
from .features import extract_clip_patch_features
from .io import load_projector, save_projector, validate_projector
from .pipeline import construct_portal, construct_portal_ctx, vision_source_stats
from .stats import pca_stats
from .targets import llm_embedding_table, llm_hidden_states

__version__ = "1.0.0"

__all__ = [
    "build_portal",
    "construct_portal",
    "construct_portal_ctx",
    "vision_source_stats",
    "pca_stats",
    "extract_clip_patch_features",
    "llm_embedding_table",
    "llm_hidden_states",
    "random_baseline",
    "make_projector_state_dict",
    "save_projector",
    "load_projector",
    "validate_projector",
]
