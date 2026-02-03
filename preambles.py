"""Legacy preamble imports.

This module exists for compatibility with legacy notebooks/scripts.
New code should import from `src/HighDimSpatial/` directly.
"""

import itertools
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.optim as optim
from scipy.interpolate import griddata
from scipy.linalg import cholesky, cho_factor, cho_solve, det, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.special import beta as B
from scipy.special import gamma
from scipy.special import kv, kvp
from scipy.stats import multivariate_normal, spearmanr
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from HighDimSpatial.config import DEFAULT_DTYPE, get_device

# Legacy global device used across notebooks
device = get_device()
