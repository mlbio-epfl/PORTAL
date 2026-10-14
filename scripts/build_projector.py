#!/usr/bin/env python
"""Construct a PORTAL projector checkpoint from the command line.

Options can be given as flags or loaded from a YAML file with --config
(flags override the file; see configs/portal.yaml).

Examples:

Default PORTAL (target = LLM input embeddings), Vicuna 1.5 7B + CLIP:

    python scripts/build_projector.py \
        --image-dir /data/coco/val2017 \
        --llm-model llava-hf/llava-1.5-7b-hf \
        --out checkpoints/portal_vicuna_clip

From a config file:

    python scripts/build_projector.py --config configs/portal.yaml

PORTAL-ctx targeting hidden states after block 4:

    python scripts/build_projector.py \
        --image-dir /data/coco/val2017 \
        --llm-model meta-llama/Llama-3.1-8B \
        --layer 4 --prompts examples/words.txt \
        --out checkpoints/portal_ctx4_llama

Random baseline (no image pool needed; dims inferred from the LLM):

    python scripts/build_projector.py --method random \
        --llm-model llava-hf/llava-1.5-7b-hf \
        --out checkpoints/random_vicuna
"""

import argparse
from pathlib import Path

from loguru import logger

from portal import (
    construct_portal,
    construct_portal_ctx,
    llm_embedding_table,
    random_baseline,
    save_projector,
)


def parse_args() -> argparse.Namespace:
    # Pre-parse --config so its values can seed the defaults below; any flag
    # passed explicitly on the command line still overrides the file.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config", default=None, help="YAML file of options (flags override it)."
    )
    config_arg, _ = pre.parse_known_args()
    config_defaults: dict = {}
    if config_arg.config:
        import yaml

        with open(config_arg.config) as f:
            config_defaults = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser(parents=[pre], description="Construct a PORTAL projector.")
    p.add_argument(
        "--method",
        choices=["portal", "random"],
        default="portal",
        help="'portal' (optionally with --layer for PORTAL-ctx) or 'random'.",
    )
    p.add_argument("--out", help="Output directory for the checkpoint.")
    p.add_argument(
        "--llm-model",
        help="HuggingFace LLM identifier (plain CausalLM or a LLaVA model).",
    )
    p.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14-336",
        help="HuggingFace CLIP vision model identifier.",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="Unlabeled image pool for vision statistics (required for PORTAL).",
    )
    p.add_argument("--k", type=int, default=512, help="PCA rank (paper default: 512).")
    p.add_argument("--n-images", type=int, default=1024, help="Images to sample.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--max-scale-ratio",
        type=float,
        default=100.0,
        help="Clamp on per-axis scale sigma_y/sigma_x; <= 0 (or null in a config) disables it.",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Target block for PORTAL-ctx (0 = input embeddings = default PORTAL).",
    )
    p.add_argument(
        "--prompts",
        default=None,
        help="Newline-delimited words/phrases file (required when --layer > 0).",
    )
    p.add_argument(
        "--config-source",
        default=None,
        help="Optional LLaVA checkpoint dir whose config.json is copied alongside.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    p.set_defaults(**config_defaults)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.llm_model:
        raise SystemExit("--llm-model is required (pass it as a flag or in --config)")
    if not args.out:
        raise SystemExit("--out is required (pass it as a flag or in --config)")

    max_scale = None if (args.max_scale_ratio is None or args.max_scale_ratio <= 0) else args.max_scale_ratio

    if args.method == "random":
        if args.image_dir or args.prompts or args.layer:
            logger.warning("--method random ignores --image-dir, --prompts, and --layer")
        vision_dim = _infer_vision_dim(args.clip_model)
        llm_dim = llm_embedding_table(args.llm_model).shape[-1]
        state_dict = random_baseline(vision_dim, llm_dim, seed=args.seed)

    elif args.layer > 0:
        if not args.image_dir:
            raise SystemExit("--image-dir is required for PORTAL")
        if not args.prompts:
            raise SystemExit("--prompts is required when --layer > 0 (PORTAL-ctx)")
        state_dict = construct_portal_ctx(
            image_dir=args.image_dir,
            llm_model=args.llm_model,
            layer=args.layer,
            prompts=args.prompts,
            clip_model=args.clip_model,
            k=args.k,
            n_images=args.n_images,
            max_scale_ratio=max_scale,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )

    else:
        if not args.image_dir:
            raise SystemExit("--image-dir is required for PORTAL")
        if args.prompts:
            logger.warning(
                "--prompts is ignored for default PORTAL (layer=0); "
                "pass --layer L>0 for PORTAL-ctx"
            )
        state_dict = construct_portal(
            image_dir=args.image_dir,
            llm_model=args.llm_model,
            clip_model=args.clip_model,
            k=args.k,
            n_images=args.n_images,
            max_scale_ratio=max_scale,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )

    out = save_projector(state_dict, Path(args.out), config_source=args.config_source)
    logger.info("done: {}", out)


def _infer_vision_dim(clip_model: str) -> int:
    """Read the CLIP vision hidden size from its config (no image pool needed)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(clip_model)
    vision_cfg = getattr(cfg, "vision_config", cfg)
    return vision_cfg.hidden_size


if __name__ == "__main__":
    main()
