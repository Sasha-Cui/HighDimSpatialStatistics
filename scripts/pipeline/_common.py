"""Shared helpers for pipeline scripts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch


def add_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


def save_tensors(path: Path, tensors: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)


def load_tensors(path: Path) -> Dict[str, Any]:
    """Load tensor-only research payloads onto CPU using Torch's restricted loader."""
    return torch.load(path, map_location="cpu", weights_only=True)
