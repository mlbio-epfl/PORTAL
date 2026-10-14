"""End-to-end PORTAL construction: features -> statistics -> projector.

Convenience wrappers around the feature, stats, and construction steps.
scripts/build_projector.py is a thin CLI over these.
"""

from pathlib import Path

from loguru import logger
from torch import Tensor

from .construct import build_portal
from .features import extract_clip_patch_features
from .stats import Stats, pca_stats
from .targets import llm_embedding_table, llm_hidden_states, load_prompts

__all__ = ["vision_source_stats", "construct_portal", "construct_portal_ctx"]


def vision_source_stats(
    image_dir: str | Path,
    clip_model: str = "openai/clip-vit-large-patch14-336",
    k: int = 512,
    n_images: int = 1024,
    batch_size: int = 8,
    seed: int = 42,
    device: str | None = None,
) -> Stats:
    """Extract CLIP patch features over an image pool and summarize them."""
    feats = extract_clip_patch_features(
        image_dir,
        clip_model=clip_model,
        n_images=n_images,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    return pca_stats(feats, k)


def construct_portal(
    image_dir: str | Path,
    llm_model: str,
    clip_model: str = "openai/clip-vit-large-patch14-336",
    k: int = 512,
    n_images: int = 1024,
    max_scale_ratio: float | None = 100.0,
    batch_size: int = 8,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Tensor]:
    """Construct the default PORTAL projector (target = LLM input embeddings).

    Needs only an unlabeled image pool and the LLM's embedding table. No paired
    image-caption data and no gradient steps.
    """
    logger.info("PORTAL: {} vision -> {} embeddings (k={})", clip_model, llm_model, k)
    source = vision_source_stats(
        image_dir,
        clip_model=clip_model,
        k=k,
        n_images=n_images,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    target = pca_stats(llm_embedding_table(llm_model), k)
    return build_portal(source, target, max_scale_ratio=max_scale_ratio)


def construct_portal_ctx(
    image_dir: str | Path,
    llm_model: str,
    layer: int,
    prompts: list[str] | str | Path,
    clip_model: str = "openai/clip-vit-large-patch14-336",
    k: int = 512,
    n_images: int = 1024,
    max_scale_ratio: float | None = 100.0,
    batch_size: int = 8,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Tensor]:
    """Construct a PORTAL-ctx-L projector targeting LLM hidden states after block L.

    `layer` > 0 picks the target block (L=0 recovers the embedding target; use
    construct_portal for that). `prompts` is a list of words/short phrases or a
    path to a newline-delimited file.
    """
    if isinstance(prompts, (str, Path)):
        prompts = load_prompts(prompts)
    logger.info("PORTAL-ctx{}: {} vision -> {} hidden states", layer, clip_model, llm_model)
    source = vision_source_stats(
        image_dir,
        clip_model=clip_model,
        k=k,
        n_images=n_images,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    states = llm_hidden_states(llm_model, prompts, layer=layer, device=device)
    target = pca_stats(states, k)
    return build_portal(source, target, max_scale_ratio=max_scale_ratio)
