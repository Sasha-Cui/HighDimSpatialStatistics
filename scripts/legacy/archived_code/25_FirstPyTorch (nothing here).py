# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/archived_code/25_FirstPyTorch (nothing here).ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Create two random tensors on the CPU
x = torch.randn(3, 3)
y = torch.randn(3, 3)

print("Tensors on CPU:")
print("x =", x)
print("y =", y)

# Move tensors to the GPU
x = x.to(device)
y = y.to(device)

print("\nTensors on GPU:")
print("x =", x)
print("y =", y)

# Perform a simple tensor operation on the GPU
z = x + y

print("\nResult of tensor addition on GPU:")
print("z =", z)

# Move the result back to the CPU (if needed)
z_cpu = z.to("cpu")

print("\nResult moved back to CPU:")
print("z_cpu =", z_cpu)

# %%
# Finally, I am able to make use of GPU. 
#
# Consequently, the next steps are the move the operations of file 26 and 27 to GPU. 

# %%
pass

# %%
# This is the old, scipy-based way to calculate.  We leave them here for sanity check.
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.linalg import cholesky, solve_triangular, det

def matern_covariance(X, length_scale=1.0, nu=1.5, sigma2=1.0):
    """
    Computes the Matérn covariance matrix given the input points and kernel parameters.
    
    Parameters:
    - X (np.ndarray): The input points (n_samples, n_features).
    - length_scale (float): Length scale parameter of the Matérn kernel.
    - nu (float): Smoothness parameter of the Matérn kernel.
    - sigma2 (float): Variance (signal variance) parameter of the Matérn kernel.
    
    Returns:
    - cov_matrix (np.ndarray): The computed Matérn covariance matrix.
    """
    # Define the Matern kernel with the given length_scale and nu
    matern_kernel = Matern(length_scale=length_scale, nu=nu)
    
    # Combine with a ConstantKernel to include sigma2
    kernel = ConstantKernel(constant_value=sigma2) * matern_kernel
    
    # Compute the covariance matrix
    cov_matrix = kernel(X)
    
    return cov_matrix

def negative_log_likelihood(y, cov_matrix):
    """
    Computes the negative log-likelihood for a zero-mean Gaussian Process.
    
    Parameters:
    - y (np.ndarray): The observation vector (n_samples,).
    - cov_matrix (np.ndarray): The covariance matrix (n_samples, n_samples).
    
    Returns:
    - nll (float): The negative log-likelihood.
    """
    # Cholesky decomposition
    L = cholesky(cov_matrix, lower=True)
    
    # Solve L * alpha = y
    alpha = solve_triangular(L, y, lower=True)
    
    # Compute the negative log-likelihood
    log_likelihood = 0.5 * np.dot(alpha, alpha)
    log_likelihood += np.sum(np.log(np.diag(L)))
    log_likelihood += 0.5 * len(y) * np.log(2 * np.pi)
    
    return log_likelihood

# Define the locations (points) in R^2
X = np.array([[1.0, 2.0],  # Location 1
              [3.0, 4.0],
             [5.0,6.0]]) # Location 2

# Define the observation vector y (assume zero-mean GP)
y = np.array([1.0, -1.0, 1.0])

# Define the Matérn kernel parameters
length_scale = 1.0
nu = 1.0
sigma2 = 1.0

# Compute the Matérn covariance matrix
cov_matrix = matern_covariance(X, length_scale=length_scale, nu=nu, sigma2=sigma2)

# Compute the negative log-likelihood
nll = negative_log_likelihood(y, cov_matrix)

# Print the covariance matrix and the negative log-likelihood
print("Covariance matrix with sigma2 included:\n", cov_matrix)
print("Negative log-likelihood:", nll)

# %%
import torch
from torch.autograd import Function
from scipy.special import kv, kvp, gamma
from torch.linalg import cholesky, solve_triangular

# Define the custom autograd function for the Bessel function of the second kind
class BesselKFunction(Function):
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
        
        return grad_input_v.sum(), grad_input_x
        

# Matern kernel function with gradient-tracking Bessel function
def matern_kernel(pairwise_distances, nu, length_scale, sigma2):
    # Ensure that nu, length_scale, and sigma2 are scalar tensors
    assert nu.dim() == 0 and length_scale.dim() == 0 and sigma2.dim() == 0
    
    # Compute the scaled distances
    scaled_distances = torch.sqrt(2 * nu) * (pairwise_distances / length_scale)
    # Clamp the values of scaled_distances to avoid extreme numbers
    scaled_distances = torch.clamp(scaled_distances, min=1e-9, max=1e6)

    
    if nu == 0.5:
        covariance_matrix = sigma2 * torch.exp(-scaled_distances)
    else:
        # Use the custom Bessel function with autograd
        bessel_term = BesselKFunction.apply(nu, scaled_distances)
        scaling_term = (2 ** (1.0 - nu)) / torch.exp(torch.lgamma(nu))
        covariance_matrix = sigma2 * scaling_term * (scaled_distances ** nu) * bessel_term
        covariance_matrix = torch.where(pairwise_distances == 0, sigma2, covariance_matrix)
    
    return covariance_matrix

# Compute the negative log-likelihood loss
def negative_log_likelihood(y, cov_matrix):
    n = y.shape[0]
    L = cholesky(cov_matrix, upper=False)
    alpha = solve_triangular(L, y.view(-1, 1), upper=False)
    log_likelihood = 0.5 * torch.sum(alpha ** 2)
    log_likelihood += torch.sum(torch.log(torch.diag(L)))
    log_likelihood += 0.5 * n * torch.log(torch.tensor(2 * torch.pi))
    return log_likelihood


# %%
# Example usage with the Matern kernel
# Define the locations (points) in R^2
X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                 [5.0,6.0]], requires_grad=True)
# Example observation vector y (assume zero-mean GP)
y = torch.tensor([1.0, -1.0, 1.0])

# %%
# Define the Matérn kernel parameters as tensors
nu = torch.tensor(1.0, requires_grad=True)
length_scale = torch.tensor(1.0, requires_grad=True)
sigma2 = torch.tensor(1.0, requires_grad=True)

# Compute the pairwise distances using torch.cdist
pairwise_distances = torch.cdist(X, X)

# Compute the Matérn covariance matrix
cov_matrix = matern_kernel(pairwise_distances, nu, length_scale, sigma2)
cov_matrix += torch.eye(cov_matrix.size(0)) * 1e-6

loss = negative_log_likelihood(y, cov_matrix)

# Perform backpropagation
loss.backward()
print("Covariance matrix:", cov_matrix)
print("loss", loss)
print("\n\n")

# Print the gradients with respect to the points, nu, length_scale, and sigma2
print("Gradient of points:\n", X.grad)
print("Gradient of nu:", nu.grad)
print("Gradient of length_scale:", length_scale.grad)
print("Gradient of sigma2:", sigma2.grad)

# %%
from torch.optim import Adam

### Data Generation
# Define the locations (points) in R^2
X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                 [5.0,6.0]], requires_grad=True)
# Example observation vector y (assume zero-mean GP)
y = torch.tensor([1.0, -1.0, 1.0])
# Compute the pairwise distances using torch.cdist
pairwise_distances = torch.cdist(X, X)

### Parameter Definition
# Define the Matérn kernel parameters as tensors
nu = torch.tensor(1.0, requires_grad=True)
length_scale = torch.tensor(1.0, requires_grad=True)
sigma2 = torch.tensor(1.0, requires_grad=True)


# Define the parameter bounds
sigma2_min, sigma2_max = 0.001, 10.0
length_scale_min, length_scale_max = 0.001, 10.0
nu_min, nu_max = 0.001, 5.0


# Assuming the previous code is already defined, including BesselKFunction, matern_kernel, and negative_log_likelihood

# Initialize the optimizer with the parameters to optimize
optimizer = Adam([nu, length_scale, sigma2], lr=0.01)  # You can adjust the learning rate as needed

# Number of optimization steps
n_steps = 10000

# Optimization loop
for step in range(n_steps):
    optimizer.zero_grad()  # Zero the gradients from the previous step
    
    # Recompute the covariance matrix with the current parameters
    cov_matrix = matern_kernel(pairwise_distances, nu, length_scale, sigma2)
    cov_matrix += torch.eye(cov_matrix.size(0)) * 1e-6  # Add jitter for numerical stability
    
    # Compute the loss (negative log-likelihood)
    loss = negative_log_likelihood(y, cov_matrix)
    
    # Backpropagate to compute the gradients
    loss.backward(retain_graph=True)  # Ensure this is only called once per loop iteration
    
    # Update the parameters
    optimizer.step()

    # Clip the parameters to stay within the specified bounds
    with torch.no_grad():
        nu.clamp_(nu_min, nu_max)
        length_scale.clamp_(length_scale_min, length_scale_max)
        sigma2.clamp_(sigma2_min, sigma2_max)
    
    # Print the loss and parameters every 100 steps for monitoring
    if step % (n_steps/10) == 0 or step == n_steps - 1:
        print(f"Step {step + 1}/{n_steps}, Loss: {loss.item()}")
        print(f"nu: {nu.item()}, length_scale: {length_scale.item()}, sigma2: {sigma2.item()}")

# After optimization, the optimal parameters are:
optimal_nu = nu.item()
optimal_length_scale = length_scale.item()
optimal_sigma2 = sigma2.item()

print("\nOptimization complete!")
print(f"Optimal nu: {optimal_nu}")
print(f"Optimal length_scale: {optimal_length_scale}")
print(f"Optimal sigma2: {optimal_sigma2}")

# %%
pass

