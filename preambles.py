# Import packages
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.linalg import cholesky, solve_triangular, det
import torch
from torch.autograd import Function
from torch.linalg import cholesky, solve_triangular
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
import sys
import random
from scipy.special import kv, kvp, gamma
from scipy.special import beta as B
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.interpolate import griddata
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import minimize
import os
from scipy.stats import spearmanr
import time, itertools
import multiprocessing as mp
from itertools import product
# # for the torch code, we, don't really need those
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import multivariate_normal
# from sklearn.gaussian_process.kernels import Matern, ConstantKernel
# from scipy.interpolate import griddata
# from scipy.linalg import cho_solve, cho_factor
# from scipy.optimize import minimize
# from scipy.special import beta as B
# from scipy.special import gamma as Gamma
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
