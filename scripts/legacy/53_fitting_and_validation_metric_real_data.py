# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/53_fitting_and_validation_metric_real_data.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Just write down the two directories.  The estimation results should come out in the output address

# %%
input_dir_base = '~/project/52_subsampled_hattie_data'
output_dir_base = '~/project/53_hattie_marginal_estimation_results'

# %%
# # This is the notebook that takes in the kernel-smoothed data and does fitting.
# 1. The nominal input is the directory to the csv's of the preprocessed data,
# 2. The real inputs are the learning rates and initialisation values (i.e. hyperparameters),
# 3. The nominal outputs in the estimated $\alpha, \nu, \sigma$
# 4. The real outputs are the validation metrics of the fitted parameters, for different hyperparameters.
# 5. Also optionally prints the losses by epochs.  By default, they are suppressed.

# %%
# ## First half, computing the estimated parameters

# %%
number_of_cycles, steps_per_batch = 5000 , 2

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
from itertools import product

# Fixed values for learning rates
alpha_lr = 0.001
sigma_lr = 0.001

# Fixed value for sigma initialization
sigma_init = 1.0

# Variable values for learning rates and initializations
nu_lr_values = [0.015, 0.0075, 0.00375]
alpha_init_values = [10, 1, 0.1, 0.01]
nu_init_values = [10, 1, 0.1]

# Generate learning_rates
learning_rates = [
    {'alpha_lr': alpha_lr, 'nu_lr': nu_lr, 'sigma_lr': sigma_lr}
    for nu_lr in nu_lr_values
]

# Generate initializations
initializations = [
    {'alpha_init': alpha_init, 'nu_init': nu_init, 'sigma_init': sigma_init}
    for alpha_init, nu_init in product(alpha_init_values, nu_init_values)
]
# Print results
print("Generated Learning Rates:")
for lr in learning_rates:
    print(lr)

print("\nGenerated Initializations:")
for init in initializations:
    print(init)

print("\nOverall Hyperparameters:")
for counter, (lr_set, init_set) in enumerate(product(learning_rates, initializations)):
    l = counter 
    print("hyperparameter ", l, lr_set, init_set)

# %%
# i=0
# print(f'{input_dir_base}/{i}')

# %%
for counter, (lr_set, init_set) in enumerate(product(learning_rates, initializations)):
    l = counter 
    print("hyperparameter ", l)
    if l < 3:
        continue
    # Define output directories for the current hyperparameter setting `l`
    optimized_dir = os.path.expanduser(f'{output_dir_base}/{l}')
    os.makedirs(optimized_dir, exist_ok=True)

    i = 0
    while os.path.exists(os.path.expanduser(f'{input_dir_base}/{i}')):
        X_groups, Y_groups = [], []
        k = 0
        
        while True: # Load all X_ and Y_groups (k) of data for each dataset (i)
            x_path = os.path.expanduser(f'{input_dir_base}/{i}/X_subsampled_{k}.csv')
            y_path = os.path.expanduser(f'{input_dir_base}/{i}/Y_subsampled_{k}.csv')
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                break
            X_groups.append(torch.tensor(pd.read_csv(x_path, header=None).values, dtype=torch.float64))
            Y_groups.append(torch.tensor(pd.read_csv(y_path, header=None).values, dtype=torch.float64))
            k += 1
        # Store hyperparameters
        hyperparameters_path = os.path.expanduser(f'{output_dir_base}/hyperparameters_{l}.csv')
        pd.DataFrame([lr_set | init_set]).to_csv(hyperparameters_path, index=False)
    
        # Optimize parameters
        optimized_params, best_params, loss_histories = optimize_marginal_parameters_in_groups(lr_set, init_set, X_groups, Y_groups, number_of_cycles, steps_per_batch, sigma_is_known = False)
        print([loss_history[-1] for loss_history in loss_histories])
        
        # Store optimized and best parameters for each feature `j`
        for j, (optimized_params_j, best_params_j) in enumerate(zip(optimized_params, best_params)):
            # Convert optimized_params_j to float values if they are tensors
            optimized_params_j = [v.item() if isinstance(v, torch.Tensor) else v for v in optimized_params_j]

            # Convert best_params_j to a flat list of float values, assuming it's a list with a single dictionary
            best_params_j = [v.item() if isinstance(v, torch.Tensor) else v for v in best_params_j[0].values()]
        
            # Store optimized parameters
            optimized_params_path = f'{optimized_dir}/optimized_parameters_dataset_{i}_feature_{j}.csv'
            os.makedirs(os.path.dirname(optimized_params_path), exist_ok=True)
            pd.DataFrame([optimized_params_j], columns=['alpha', 'nu', 'sigma']).to_csv(optimized_params_path, index=False)

            # Store best parameters
            best_params_path = f'{optimized_dir}/best_parameters_dataset_{i}_feature_{j}.csv'
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            pd.DataFrame([best_params_j], columns=['alpha', 'nu', 'sigma']).to_csv(best_params_path, index=False)

            # Store full loss histories
            loss_histories_path = f'{optimized_dir}/loss_histories_dataset_{i}_feature_{j}.csv'
            os.makedirs(os.path.dirname(loss_histories_path), exist_ok=True)
            if j < len(loss_histories):  # Ensure j is within range
                pd.DataFrame({'loss': list(loss_histories[j])}).to_csv(loss_histories_path, index=False)
        i += 1  # Move to the next dataset index for the current hyperparameter setting
        if i == 2: 
            break
print("Optimization process completed.")

# %%
# # debugging codes

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

l = 0
init_set = initializations[0]
lr_set = learning_rates[0]

# Define output directories for the current hyperparameter setting `l`
optimized_dir = os.path.expanduser(f'{output_dir_base}/{l}')
os.makedirs(optimized_dir, exist_ok=True)

i = 0
X_groups, Y_groups = [], []
k = 0
while True: # Load all X_ and Y_groups (k) of data for each dataset (i)
    x_path = os.path.expanduser(f'{input_dir_base}/{i}/X_subsampled_{k}.csv')
    y_path = os.path.expanduser(f'{input_dir_base}/{i}/Y_subsampled_{k}.csv')
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        break
    X_groups.append(torch.tensor(pd.read_csv(x_path, header=None).values, dtype=torch.float64))
    Y_groups.append(torch.tensor(pd.read_csv(y_path, header=None).values, dtype=torch.float64))
    k += 1

alpha_i = torch.tensor(init_set['alpha_init'], dtype=torch.float64, device=device).requires_grad_(True)
nu_i = torch.tensor(init_set['nu_init'], dtype=torch.float64, device=device).requires_grad_(True)
sigma_i = torch.tensor(1.0, dtype=torch.float64, device=device).requires_grad_(False)

# Optimize parameters
optimized_params, best_params, loss_histories = optimize_marginal_parameters_in_groups(lr_set, init_set, X_groups, Y_groups, number_of_cycles, steps_per_batch)

# %%
# def adjust_matrix_with_nugget(K, nugget):
#     """
#     Adjust a symmetric matrix K by adding a nugget if necessary.

#     Parameters:
#     - K (torch.Tensor): The input symmetric matrix, already on the desired device.
#     - nugget (float): The size of the additional nugget to be added after ensuring positive definiteness.

#     Returns:
#     - K_adjusted (torch.Tensor): The adjusted matrix, kept on the same device.
#     """
#     # Ensure K is symmetric
#     assert torch.allclose(K, K.T), "Matrix K must be symmetric"

#     # Determine the device
#     device = K.device

#     # Calculate eigenvalues
#     eigenvalues = torch.linalg.eigvalsh(K)  # Use eigvalsh for symmetric matrices
#     smallest_eigenvalue = eigenvalues.min().item()

#     # Adjust the matrix if the smallest eigenvalue is negative
#     if smallest_eigenvalue < 0:
#         K = K + (-smallest_eigenvalue) * torch.eye(K.shape[0], device=device, dtype=K.dtype)

#     # Add the additional nugget
#     K = K + nugget * torch.eye(K.shape[0], device=device, dtype=K.dtype)

#     return K

# %%
# X_batch = X_groups[9]
# Y_batch = Y_groups[9]

# K = matern_kernel(torch.cdist(X_batch, X_batch), alpha_i, nu_i, sigma_i)
# K += torch.eye(K.size(0), device=device) * 1e-9
# K_approx_old = approx_matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
# K_approx = adjust_matrix_with_nugget(K_approx, nugget=1e-9)

# print(torch.linalg.norm(K-K_approx).item(), torch.linalg.norm(K).item(), torch.linalg.norm(K_approx).item(), ((torch.linalg.norm(K-K_approx)/torch.linalg.norm(K)).item())**2)

# %%
# # Compute eigenvalues
# eigenvalues1 = torch.linalg.eigvals(K.detach())
# eigenvalues2 = torch.linalg.eigvals(K_approx.detach())

# # Extract real parts
# real_parts1 = torch.sort(eigenvalues1.real).values.numpy()
# real_parts2 = torch.sort(eigenvalues2.real).values.numpy()

# # Plot histograms
# plt.figure(figsize=(10, 5))
# plt.hist(real_parts2-real_parts1, bins=50, alpha=0.7, label='K_approx Real Parts - K Real Parts')
# plt.xlabel("Differences Real Part of Eigenvalues")
# plt.ylabel("Frequency")
# plt.title("Histogram of Differences of Real Parts of Eigenvalues")
# plt.legend()
# plt.show()

# %%
# real_parts2

# %%
# real_parts1

# %%
# # The verdict:  this approximated K, while being close in the relative squared forbenius norm, can be caused to have negative eigenvalues.  There are a few ways to fix this. 
#
# 1. Use the original, slow version.  This could take a long time to finish.
# 2. Add more to the diagonal of the approximation.
# 3. Try with a different initialisation.

