"""Legacy fitting functions shim.

This module preserves the original API used by notebooks, but delegates
implementation to the cleaned `HighDimSpatial` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from HighDimSpatial.config import get_device
from HighDimSpatial.fitting.cross import optimize_cross_parameters, optimize_cross_parameters_in_groups
from HighDimSpatial.fitting.marginal import optimize_marginal_parameters, optimize_marginal_parameters_in_groups


device = get_device()

__all__ = [
    "device",
    "optimize_marginal_parameters",
    "optimize_marginal_parameters_in_groups",
    "optimize_cross_parameters_in_groups",
    "optimize_cross_parameters",
]
