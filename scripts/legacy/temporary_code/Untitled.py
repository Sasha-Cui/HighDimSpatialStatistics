# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/Untitled.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import jupyterlab
print(jupyterlab.__version__)

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
alpha_values = [10, 2, 0.3]
nu_values = [2.4, 0.6, 0.15]
sigma_values = [5, 1, 0.2]
optimized_marginal_params = [
    torch.tensor([alpha, nu, sigma], dtype=torch.float64)
    for alpha, nu, sigma in zip(alpha_values, nu_values, sigma_values)
]
# Extract the optimized alpha, nu, and sigma from the list
alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)

p = 3
Delta_A = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
Delta_B = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
rho_A = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
rho_B = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
rho_V = torch.tensor(-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
W = torch.full((p,), torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)

alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)


# Convert to NumPy arrays
alpha_matrix_np = alpha_matrix.cpu().numpy() if alpha_matrix.is_cuda else alpha_matrix.detach().numpy()
nu_matrix_np = nu_matrix.cpu().numpy() if nu_matrix.is_cuda else nu_matrix.detach().numpy()
sigma_matrix_np = sigma_matrix.cpu().numpy() if sigma_matrix.is_cuda else sigma_matrix.detach().numpy()

# Display matrices in a readable form using pandas DataFrames
pd.options.display.float_format = '{:.3f}'.format

print("Inhomogeneous Parameterisation Example")
print("Alpha Matrix:")
print(pd.DataFrame(alpha_matrix_np))

print("\nNu Matrix:")
print(pd.DataFrame(nu_matrix_np))

print("\nSigma Matrix:")
print(pd.DataFrame(sigma_matrix_np))

# %%
pass

