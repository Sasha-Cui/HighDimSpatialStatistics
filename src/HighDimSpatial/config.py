"""Configuration utilities for HighDimSpatial.

All paths are repo-relative by default and can be overridden via environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


def get_repo_root() -> Path:
    """Return the repository root path.

    Resolution order:
    1) $HIGHDIMSPATIAL_ROOT if set
    2) inferred from this file location
    """
    env_root = os.getenv("HIGHDIMSPATIAL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the base data directory.

    Resolution order:
    1) $HIGHDIMSPATIAL_DATA_DIR if set
    2) <repo_root>/data
    """
    env_data = os.getenv("HIGHDIMSPATIAL_DATA_DIR")
    if env_data:
        return Path(env_data).expanduser().resolve()
    return get_repo_root() / "data"


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return torch device (CUDA if available and preferred)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_data_path(*parts: str, must_exist: bool = False) -> Path:
    """Resolve a path under the data directory.

    Args:
        *parts: Path components under data dir.
        must_exist: If True, raises FileNotFoundError when not found.
    """
    path = get_data_dir().joinpath(*parts)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Missing data path: {path}")
    return path


DEFAULT_DTYPE = torch.float64
