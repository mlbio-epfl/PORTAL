"""Save and load PORTAL projectors in the LLaVA checkpoint format."""

import shutil
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor

from .construct import PROJECTOR_KEYS

__all__ = ["save_projector", "load_projector", "validate_projector", "PROJECTOR_KEYS"]

_CHECKPOINT_NAME = "mm_projector_f.bin"


def validate_projector(state_dict: dict[str, Tensor]) -> None:
    """Check the projector has the expected keys and mutually consistent shapes."""
    for key in PROJECTOR_KEYS:
        if key not in state_dict:
            raise KeyError(f"missing projector key: {key}")

    w1 = state_dict[PROJECTOR_KEYS[0]]
    llm_dim, vision_dim = w1.shape
    expected = {
        PROJECTOR_KEYS[0]: (llm_dim, vision_dim),
        PROJECTOR_KEYS[1]: (llm_dim,),
        PROJECTOR_KEYS[2]: (llm_dim, llm_dim),
        PROJECTOR_KEYS[3]: (llm_dim,),
    }
    for key, shape in expected.items():
        actual = tuple(state_dict[key].shape)
        if actual != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {actual}")


def save_projector(
    state_dict: dict[str, Tensor],
    out_dir: str | Path,
    config_source: str | Path | None = None,
) -> Path:
    """Save a projector as `<out_dir>/mm_projector_f.bin` and return its path.

    If `config_source` (a directory or config.json) is given, its config is
    copied next to the checkpoint, handy when wiring PORTAL into a LLaVA model.
    """
    validate_projector(state_dict)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    bin_path = out_dir / _CHECKPOINT_NAME
    torch.save(state_dict, bin_path)

    if config_source is not None:
        src = Path(config_source)
        if src.is_dir():
            src = src / "config.json"
        if src.exists():
            shutil.copy2(src, out_dir / "config.json")

    logger.info("saved projector to {}", bin_path)
    return bin_path


def load_projector(path: str | Path) -> dict[str, Tensor]:
    """Load a projector checkpoint (a directory or a direct .bin path)."""
    path = Path(path)
    if path.is_dir():
        path = path / _CHECKPOINT_NAME
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    validate_projector(state_dict)
    return state_dict
