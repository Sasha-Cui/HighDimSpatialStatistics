# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/archived_code/29_PositiveDefinite (obsolete).ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # The _PositiveDefinite() function in pytorch is problematic.

# %%
# Import packages
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.linalg import cholesky, solve_triangular, det
import torch
from torch.autograd import Function
from torch.linalg import cholesky, solve_triangular
import torch.optim as optim
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

def is_positive_definite(matrix):
    """Check if a matrix is positive definite."""
    try:
        torch.linalg.cholesky(matrix)
        return True
    except RuntimeError:
        return False

# Define the custom autograd function for the Bessel function of the second kind
class BesselKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, x):
        # Store v and x for the backward pass
        ctx.save_for_backward(v, x)
        
        # Convert tensors to numpy for SciPy compatibility
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        
        # Compute the Bessel function using SciPy
        output = torch.tensor(kv(v_np, x_np), dtype=torch.float32)
        
        # Return the output as a tensor
        return output.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        v, x = ctx.saved_tensors
        
        # Convert tensors to numpy for gradient calculations
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        
        # Numerical derivative with respect to x
        epsilon_x = 1e-7
        grad_x = (kv(v_np, x_np + epsilon_x) - kv(v_np, x_np - epsilon_x)) / (2 * epsilon_x)
        grad_x = torch.tensor(grad_x, dtype=torch.float32).to(x.device)
        
        # Derivative with respect to v (using kvp, which gives the derivative of kv with respect to v)
        grad_v = torch.tensor(kvp(v_np, x_np), dtype=torch.float32).to(v.device)
        
        # Multiply by the incoming gradient (chain rule)
        grad_input_x = grad_output * grad_x
        grad_input_v = grad_output * grad_v
        
        return grad_input_v, grad_input_x

def matern_kernel(pairwise_distances, nu, length_scale, sigma2, epsilon=1e-6):
    """
    Computes the Matérn covariance matrix with support for broadcasting over nu, length_scale, and sigma2.

    Parameters:
    - pairwise_distances (torch.Tensor): Pairwise distances, shape (n_locations, n_locations) or broadcasted to (p, p, n_locations, n_locations).
    - nu (torch.Tensor): Smoothness parameter, can be broadcasted (e.g., shape (p, p, 1, 1)).
    - length_scale (torch.Tensor): Length scale parameter, can be broadcasted (e.g., shape (p, p, 1, 1)).
    - sigma2 (torch.Tensor): Variance parameter, can be broadcasted (e.g., shape (p, p, 1, 1)).
    - epsilon (float): A small perturbation to ensure nu != 0.5.

    Returns:
    - covariance_matrix (torch.Tensor): The computed Matérn covariance matrix, with shape depending on input broadcasting.
    """

    # Add a tiny perturbation to nu if it's exactly 0.5
    nu = torch.where(nu == 0.5, nu + epsilon, nu)

    # Compute the scaled distances using broadcasting
    scaled_distances = torch.sqrt(2 * nu) * (pairwise_distances / length_scale)
    
    # Clamp the values of scaled_distances to avoid extreme numbers
    scaled_distances = torch.clamp(scaled_distances, min=1e-9, max=1e6)

    # Use the custom Bessel function with autograd
    bessel_term = BesselKFunction.apply(nu, scaled_distances)
    scaling_term = (2 ** (1.0 - nu)) / torch.exp(torch.lgamma(nu))
    covariance_matrix = sigma2 * scaling_term * (scaled_distances ** nu) * bessel_term
    
    # Set diagonal elements where pairwise_distances == 0 to sigma2
    covariance_matrix = torch.where(pairwise_distances == 0, sigma2, covariance_matrix)
    
    return covariance_matrix

# Now that we have the alpha, nu, and sigma matrices, we want to define a function 
# Input are those three matrices, as well as X, the matrix of locations
# Output is the matern covariance matrix, computed by the function 
# matern_kernel(pairwise_distances, nu, length_scale, sigma2)
def compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X):
    """
    Computes the Matérn covariance matrix using the provided alpha, nu, and sigma matrices.

    Parameters:
    - alpha_matrix (torch.Tensor): The alpha matrix of shape (p, p).
    - nu_matrix (torch.Tensor): The nu matrix of shape (p, p).
    - sigma_matrix (torch.Tensor): The sigma matrix of shape (p, p).
    - X (torch.Tensor): The matrix of locations of shape (n_locations, dimensions).

    Returns:
    - K (torch.Tensor): The Matérn covariance matrix of shape (n_locations * p, n_locations * p).
    """
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    # Compute pairwise distances between locations (n_locations, n_locations)
    pairwise_distances = torch.cdist(X, X)
    
    # Expand pairwise distances to (p, p, n_locations, n_locations) for broadcasting
    pairwise_distances_expanded = pairwise_distances.unsqueeze(0).unsqueeze(0).expand(p, p, n_locations, n_locations)
    
    # Ensure the correct shapes for broadcasting
    nu_expanded = nu_matrix.unsqueeze(-1).unsqueeze(-1)  # Shape: (p, p, 1, 1)
    alpha_expanded = alpha_matrix.unsqueeze(-1).unsqueeze(-1)  # Shape: (p, p, 1, 1)
    sigma_expanded = sigma_matrix.unsqueeze(-1).unsqueeze(-1)  # Shape: (p, p, 1, 1)

    # Compute the covariance matrix using broadcasting
    K_blocks = matern_kernel(
        pairwise_distances_expanded, 
        nu_expanded,  
        alpha_expanded,  
        sigma_expanded
    )
    
    # Reshape to create the block covariance matrix (n_locations * p, n_locations * p)
    K = K_blocks.permute(0, 2, 1, 3).reshape(p * n_locations, p * n_locations)
    
    return K

# Now simulate in pytorch number_of_locations locations
def simulate_locations(number_of_locations, dimensions=2, range_min=-3.0, range_max=3.0):
    """
    Simulates random locations within a specified range.

    Parameters:
    - number_of_locations (int): The number of locations to simulate.
    - dimensions (int): The number of dimensions for each location (e.g., 2 for 2D, 3 for 3D).
    - range_min (float): The minimum value for the location coordinates.
    - range_max (float): The maximum value for the location coordinates.

    Returns:
    - locations (torch.Tensor): A tensor of shape (number_of_locations, dimensions) containing the simulated locations.
    """
    # Generate random locations within the specified range
    locations = torch.FloatTensor(number_of_locations, dimensions).uniform_(range_min, range_max)
    return locations


# Now that we have X and K,  simulate p-variate data sampled at locations in X with a Gaussian process along Normal(0,K) 
def simulate_gp_data(X, K):
    """
    Simulates a p-variate dataset sampled at locations in X with a Gaussian process.
    
    Parameters:
    - X (torch.Tensor): The matrix of locations of shape (n_locations, dimensions).
    - K (torch.Tensor): The covariance matrix computed using the Matérn kernel of shape (n_locations * p, n_locations * p).

    Returns:
    - Y (torch.Tensor): The simulated dataset of shape (n_locations, p).
    """
    # Get the number of locations and the dimensionality of the data
    n_locations = X.size(0)
    p = K.size(0) // n_locations

    # Sample from a multivariate normal distribution with mean 0 and covariance K
    mean = torch.zeros(K.size(0), dtype=torch.float32)


    ####    ####    ####    ####    ####
    try:
        L = torch.linalg.cholesky(K)
        print("Covariance matrix is positive definite.")
    except RuntimeError:
        print("Covariance matrix is not positive definite.")
        
    # Perform eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(K)
    
    # Separate the real and imaginary parts (if necessary)
    eigenvalues_real = eigenvalues.real
    
    # Sort the eigenvalues and the corresponding eigenvectors
    sorted_indices = torch.argsort(eigenvalues_real)
    sorted_eigenvalues = eigenvalues_real[sorted_indices]
    print("Smallest Eigenvalue:")
    print(sorted_eigenvalues[0])
    print("Largest Eigenvalue:")
    print(sorted_eigenvalues[-1])
    print(K)

    ####    ####    ####    ####    ####    
    Y = torch.distributions.MultivariateNormal(mean, covariance_matrix=K).rsample()

    # Reshape the output to have shape (n_locations, p)
    Y = Y.view(n_locations, p)

    return Y

# The particular setting in Genton's paper, with p=3
def Genton_parametrisation():    
    if p!= 3:
        print("error")
    true_nu1_value, true_a1_value, true_sigma_1_value, true_nu2_value, true_a2_value, \
    true_sigma_2_value, true_nu3_value, true_a3_value, true_sigma_3_value, true_nu12_value, \
    true_a12_value, true_sigma_12_value, true_nu13_value, true_a13_value, true_sigma_13_value, \
    true_nu23_value, true_a23_value, true_sigma_23_value = [
        1.2, 0.01, 1.0, 0.6, 0.02, 1.0, 0.3, 0.03, 1.0, 1.093, 0.0205, -0.286, 
        1.092, 0.0263, -0.181, 0.990, 0.0282, 0.274
    ]
    alpha_matrix = torch.tensor([
        [true_a1_value,  true_a12_value, true_a13_value],
        [true_a12_value, true_a2_value,  true_a23_value],
        [true_a13_value, true_a23_value, true_a3_value]
    ])
    
    nu_matrix = torch.tensor([
        [true_nu1_value, true_nu12_value, true_nu13_value],
        [true_nu12_value, true_nu2_value, true_nu23_value],
        [true_nu13_value, true_nu23_value, true_nu3_value]
    ])
    
    sigma_matrix = torch.tensor([
        [true_sigma_1_value,  true_sigma_12_value, true_sigma_13_value],
        [true_sigma_12_value, true_sigma_2_value,  true_sigma_23_value],
        [true_sigma_13_value, true_sigma_23_value, true_sigma_3_value]
    ])
    X = simulate_locations(number_of_locations, dims)
    Y = simulate_gp_data(X, compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)).detach()
    return X,Y
    

# # Global parameters:
number_of_groups = 10 # divide the data set into smaller ones, to make fitting easier.
locations_per_group = 20 # how many locations to observe per group
number_of_locations = number_of_groups * locations_per_group # total locations
dims = 2  # 2D spatial
p =  3 # how many features 

# %%
true_nu1_value, true_a1_value, true_sigma_1_value, true_nu2_value, true_a2_value, \
true_sigma_2_value, true_nu3_value, true_a3_value, true_sigma_3_value, true_nu12_value, \
true_a12_value, true_sigma_12_value, true_nu13_value, true_a13_value, true_sigma_13_value, \
true_nu23_value, true_a23_value, true_sigma_23_value = [
    1.2, 0.01, 1.0, 0.6, 0.02, 1.0, 0.3, 0.03, 1.0, 1.093, 0.0205, -0.286, 
    1.092, 0.0263, -0.181, 0.990, 0.0282, 0.274
]
alpha_matrix = torch.tensor([
    [true_a1_value,  true_a12_value, true_a13_value],
    [true_a12_value, true_a2_value,  true_a23_value],
    [true_a13_value, true_a23_value, true_a3_value]
])

nu_matrix = torch.tensor([
    [true_nu1_value, true_nu12_value, true_nu13_value],
    [true_nu12_value, true_nu2_value, true_nu23_value],
    [true_nu13_value, true_nu23_value, true_nu3_value]
])

sigma_matrix = torch.tensor([
    [true_sigma_1_value,  true_sigma_12_value, true_sigma_13_value],
    [true_sigma_12_value, true_sigma_2_value,  true_sigma_23_value],
    [true_sigma_13_value, true_sigma_23_value, true_sigma_3_value]
])
X = simulate_locations(number_of_locations, dims)
Y = simulate_gp_data(X, compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)).detach()

# %%
true_nu1_value, true_a1_value, true_sigma_1_value, true_nu2_value, true_a2_value, \
true_sigma_2_value, true_nu3_value, true_a3_value, true_sigma_3_value, true_nu12_value, \
true_a12_value, true_sigma_12_value, true_nu13_value, true_a13_value, true_sigma_13_value, \
true_nu23_value, true_a23_value, true_sigma_23_value = [
    1.2, 0.01, 1.0, 0.6, 0.02, 1.0, 0.3, 0.03, 1.0, 1.093, 0.0205, -0.286, 
    1.092, 0.0263, -0.181, 0.990, 0.0282, 0.274
]
alpha_matrix = torch.tensor([
    [true_a1_value,  true_a12_value, true_a13_value],
    [true_a12_value, true_a2_value,  true_a23_value],
    [true_a13_value, true_a23_value, true_a3_value]
])

nu_matrix = torch.tensor([
    [true_nu1_value, true_nu12_value, true_nu13_value],
    [true_nu12_value, true_nu2_value, true_nu23_value],
    [true_nu13_value, true_nu23_value, true_nu3_value]
])

sigma_matrix = torch.tensor([
    [true_sigma_1_value,  true_sigma_12_value, true_sigma_13_value],
    [true_sigma_12_value, true_sigma_2_value,  true_sigma_23_value],
    [true_sigma_13_value, true_sigma_23_value, true_sigma_3_value]
])
X = simulate_locations(number_of_locations, dims)
K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
# Get the number of locations and the dimensionality of the data

# Sample from a multivariate normal distribution with mean 0 and covariance K
mean = torch.zeros(K.size(0), dtype=torch.float32)
K=(K+K.mT)/2

Y = torch.distributions.MultivariateNormal(mean, covariance_matrix=K).rsample()

# %%
# # The error message suggests that the matrix $K$ is not Positive Definite.  Let us check whether it is.  There are three checks
# ## 1.  Whether Cholesky Decomposition is possible.
# ## 2.  Whether all eigenvalues are real and are positive.
# ## 3.  Whether the matrix is symmetric.

# %%
K

# %%
try:
    L = torch.linalg.cholesky(K)
    print("Covariance matrix is positive definite.")
except RuntimeError:
    print("Covariance matrix is not positive definite.")

# %%
# Perform eigendecomposition
eigenvalues, eigenvectors = torch.linalg.eig(K)

# Separate the real and imaginary parts (if necessary)
eigenvalues_real = eigenvalues.real

# Sort the eigenvalues and the corresponding eigenvectors
sorted_indices = torch.argsort(eigenvalues_real)
sorted_eigenvalues = eigenvalues_real[sorted_indices]
print("Smallest Eigenvalue:")
print(sorted_eigenvalues[0])
print("Largest Eigenvalue:")
print(sorted_eigenvalues[-1])

# %%
# All eigenvalues are real?
torch.all(torch.abs(torch.imag(eigenvalues)) < 1e-6)

# %%
# Is the matrix symmetric?
torch.isclose(K, K.mT, atol=1e-6).all(-2).all(-1)

# %%
torch.norm(K-K.mT)

# %%
print(K-K.mT)

# %%
H = K-K.mT
# Find the indices of non-zero elements
non_zero_indices = torch.nonzero(H, as_tuple=True)
non_zero_values = H[non_zero_indices]
# Combine indices and values, then sort them by value in descending order
non_zero_entries = list(zip(zip(*non_zero_indices), non_zero_values))
sorted_entries = sorted(non_zero_entries, key=lambda x: x[1], reverse=True)

# Print the sorted indices and their corresponding values
for idx, value in sorted_entries:
    print(f"Index: {idx}, Value: {value.item()}")

# %%
# In this implementation of https://github.com/pytorch/pytorch/blob/main/torch/distributions/constraints.py torch.distributions, there is a check for the cov matrix to be Positive Definite.  Before proceeding with the actual cholesky decomposition, the code first checks for symmetry.  The actual code is
#
#
#
# class _Symmetric(_Square):
#     """
#     Constrain to Symmetric square matrices.
#     """
#
#     def check(self, value):
#         square_check = super().check(value)
#         if not square_check.all():
#             return square_check
#         return torch.isclose(value, value.mT, atol=1e-6).all(-2).all(-1)
#
#
# class _PositiveSemidefinite(_Symmetric):
#     """
#     Constrain to positive-semidefinite matrices.
#     """
#
#     def check(self, value):
#         sym_check = super().check(value)
#         if not sym_check.all():
#             return sym_check
#         return torch.linalg.eigvalsh(value).ge(0).all(-1)
#
#
# This is the reason that multinomial distribution code in torch kept annoucing that $K$ is fails the constraint.  To keep the code running, 

# %%
pass

# %%
pass

# %%
pass

