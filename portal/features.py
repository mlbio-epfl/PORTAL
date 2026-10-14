"""Vision-side (source) features for PORTAL.

PORTAL summarizes the vision encoder by the distribution of its per-patch
tokens over an *unlabeled* image pool. No captions and no pairing are used; the
images only serve to estimate second-order statistics of the vision features.
"""

from pathlib import Path

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

__all__ = ["extract_clip_patch_features"]


def extract_clip_patch_features(
    image_dir: str | Path,
    clip_model: str = "openai/clip-vit-large-patch14-336",
    n_images: int = 1024,
    batch_size: int = 8,
    layer: int = -2,
    seed: int = 42,
    device: str | None = None,
) -> Tensor:
    """Per-patch CLIP features over an image pool (the PORTAL source).

    Reads patch tokens from the penultimate layer (layer=-2) and drops the CLS
    token, matching LLaVA's mm_vision_select_layer=-2 / select_feature=patch. A
    generic pool such as COCO val2017 is enough (paper Sec. 5.3). Returns
    (n_images, num_patches, d_vision) on CPU.
    """
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = Path(image_dir)
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = sorted({p for pat in patterns for p in image_dir.glob(pat)})
    if not paths:
        raise FileNotFoundError(f"no .jpg/.jpeg/.png images found under {image_dir}")

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(paths), generator=gen)[:n_images]
    paths = [paths[i] for i in perm]
    logger.info("extracting CLIP patch features from {} images", len(paths))

    processor = CLIPImageProcessor.from_pretrained(clip_model)
    model = CLIPVisionModel.from_pretrained(clip_model).to(device).eval()

    feats: list[Tensor] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP patches"):
        images = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        pixel_values = processor(images=images, return_tensors="pt")[
            "pixel_values"
        ].to(device)
        with torch.no_grad():
            out = model(pixel_values=pixel_values, output_hidden_states=True)
        patch = out.hidden_states[layer][:, 1:, :]  # drop CLS token
        feats.append(patch.float().cpu())

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    features = torch.cat(feats, dim=0)
    logger.info("vision patch features: {}", tuple(features.shape))
    return features
