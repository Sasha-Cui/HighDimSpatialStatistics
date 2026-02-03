"""Torch utility helpers."""
from __future__ import annotations

import random
from typing import Iterable, Optional

import numpy as np
import torch

from HighDimSpatial.config import DEFAULT_DTYPE, get_device


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Move tensor to target device if needed."""
    target = device or get_device()
    if tensor.device != target:
        return tensor.to(target)
    return tensor


def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """Ensure tensor dtype."""
    if tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor
