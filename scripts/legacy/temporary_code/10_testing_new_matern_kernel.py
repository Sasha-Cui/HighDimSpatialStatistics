# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/10_testing_new_matern_kernel.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Compare three ways of computing marginal Matern Kernel Matrix

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
# Define a threshold for the relative squared Frobenius norm
threshold = 1e-3

# Generate a simple test case (e.g., 5 points in 2D space) and move it to the device
X = torch.randn(5, 2, device=device, dtype=torch.float64)

# Define test parameters for the Matérn kernel as tensors and move them to the device
nu_i = torch.tensor(1.5, dtype=torch.float64, device=device)
alpha_i = torch.tensor(1.0, dtype=torch.float64, device=device)
sigma_i = torch.tensor(10.0, dtype=torch.float64, device=device)


# Compute the ground truth matrix using matern_kernel
K_true = matern_kernel(torch.cdist(X, X), nu_i, alpha_i, sigma_i)

# Compute the approximate matrix using approx_matern_kernel_marginal
K_approx = approx_matern_kernel_marginal(X, nu_i, alpha_i, sigma_i)

# Compute the approximate matrix using approx_matern_kernel_marginal_old
K_approx_old = approx_matern_kernel_marginal_old(X, nu_i, alpha_i, sigma_i)

# Function to compute relative squared Frobenius norm
def relative_frobenius_norm(K_approx, K_true):
    return torch.norm(K_approx - K_true, 'fro')**2 / torch.norm(K_true, 'fro')**2

# Compare the two approximate methods to the ground truth
relative_norm_approx = relative_frobenius_norm(K_approx, K_true)
relative_norm_approx_old = relative_frobenius_norm(K_approx_old, K_true)

# Output success or failure based on the threshold
if relative_norm_approx > threshold:
    print(f"Failure: approx_matern_kernel_marginal exceeds the threshold. Relative norm: {relative_norm_approx}")
else:
    print(f"Success: approx_matern_kernel_marginal is within the threshold. Relative norm: {relative_norm_approx}")

if relative_norm_approx_old > threshold:
    print(f"Failure: approx_matern_kernel_marginal_old exceeds the threshold. Relative norm: {relative_norm_approx_old}")
else:
    print(f"Success: approx_matern_kernel_marginal_old is within the threshold. Relative norm: {relative_norm_approx_old}")

# %%
K_approx_old

# %%
K_approx

# %%
K_true

# %%
pass

# %%
pass

# %%
pass

