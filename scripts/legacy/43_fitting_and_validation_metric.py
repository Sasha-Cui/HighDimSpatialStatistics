# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/43_fitting_and_validation_metric.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # This is the notebook that takes in the kernel-smoothed data and does fitting.
# 1. The nominal input is the directory to the csv's of the preprocessed data,
# 2. The real inputs are the learning rates and initialisation values (i.e. hyperparameters),
# 3. The nominal outputs in the estimated $\alpha, \nu, \sigma$
# 4. The real outputs are the validation metrics of the fitted parameters, for different hyperparameters.
# 5. Also optionally prints the losses by epochs.  By default, they are suppressed.

# %%
# ## First half, computing the estimated parameters for each hyperparameter setting.

# %%
# We run through different hyperparameters through 
#
#     for lr_set, init_set in itertools.product(learning_rates, initializations):
#
# In the l-th run, the nominal inputs are X_groups, Y_groups, lists of torch tensors that are to be read in from
#
#     '~/project/42_subsampled_synthetic_data/{i}/X_subsampled_{k}.csv'
#     '~/project/42_subsampled_synthetic_data/{i}/Y_subsampled_{k}.csv'
#
# Here, k is the list index of X_groups and Y_groups.  l is the index for different hyperparameters.  i is the index for different data sets.  j is the index for different features. 
#
# Before we do anything else, we store the hyperparameters in 
#
#     '~/project/43_estimation_results/hyperparameters_{l}.csv'
#     
# The nominal outputs are the estimated parameters obtained by the function
#
#     optimized_params, loss_histories = optimize_marginal_parameters_in_groups(lr_set, init_set, X_groups, Y_groups,number_of_cycles = 20, print_early_stopping_epochs = False)
#     
# and the estimation results are then stored in
#     
#     '~/project/43_estimation_results/{l}/optimized_parameters_{i}_{j}.csv'
#     '~/project/43_estimation_results/{l}/best_parameters_{i}_{j}.csv'
#
#
# We also store loss_histories over the training process in
#     
#     '~/project/43_estimation_results/{l}/losses_{i}_{j}/'
#
#
# The code starts with
#
#     input_dir_base = '~/project/42_subsampled_synthetic_data'
#     output_dir_base = '~/project/43_estimation_results'
#
# # Define different sets of initial learning rates and initialization values
#     learning_rates = [
#     {'alpha_lr': 0.00005, 'nu_lr': 0.001, 'sigma_lr': 0.01},
#     {'alpha_lr': 0.0003, 'nu_lr': 0.008, 'sigma_lr': 0.25},
#     {'alpha_lr': 0.0001, 'nu_lr': 0.005, 'sigma_lr': 0.1}
#     ]
#     initializations = [
#     {'alpha_init': 0.005, 'nu_init': 1.0, 'sigma_init': 1.0},
#     {'alpha_init': 0.01, 'nu_init': 1.0, 'sigma_init': 1.0},
#     {'alpha_init': 0.05, 'nu_init': 1.0, 'sigma_init': 1.0},
#     {'alpha_init': 0.1, 'nu_init': 1.0, 'sigma_init': 1.0},
#     {'alpha_init': 0.2, 'nu_init': 1.0, 'sigma_init': 1.0}
#     ]

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
number_of_cycles, steps_per_batch = 1000 , 2

# Directories
input_dir_base = '~/project/42_subsampled_synthetic_data'
output_dir_base = '~/project/43_estimation_results'

# %%
import numpy as np
from itertools import product

import numpy as np

# Base values for each hyperparameter
base_hyperparameters = {
    'alpha_lr': 0.002,
    'nu_lr': 0.03,
    'sigma_lr': 0.0001,
    'alpha_init': 0.2,  # fixed
    'nu_init': 3.0,       # fixed
    'sigma_init': 1.0     # fixed
}

# Fixed initializations
initializations = [{
    'alpha_init': base_hyperparameters['alpha_init'],
    'nu_init': base_hyperparameters['nu_init'],
    'sigma_init': base_hyperparameters['sigma_init']
}]

# Define the number of multiplicative steps
k = 3  # power range from 10^0 to 10^(k-1)
factors = [2 ** i for i in range(k)]

# Generate multiplicative combinations for each learning rate
alpha_lr_values = [base_hyperparameters['alpha_lr'] / factor for factor in factors]
nu_lr_values = [base_hyperparameters['nu_lr']/ factor for factor in factors]
sigma_lr_values = [base_hyperparameters['sigma_lr']]

# Combine into learning_rates dictionary format
learning_rates = [
    {'alpha_lr': alpha, 'nu_lr': nu, 'sigma_lr': sigma}
    for alpha in alpha_lr_values
    for nu in nu_lr_values
    for sigma in sigma_lr_values
]

# Now, learning_rates contains all combinations of multiplicative values of alpha_lr, nu_lr, and sigma_lr


# Print the generated sets
print("\nInitializations:")
print(initializations, "\n")
for learning_rate in learning_rates:
    print("Learning Rates:", learning_rate)

# %%
i=0
print(f'{input_dir_base}/{i}')

# %%
for counter, (lr_set, init_set) in enumerate(itertools.product(learning_rates, initializations)):
    l = counter 
    print("hyperparameter ", l)
    if l <7:
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
            X_groups.append(torch.tensor(pd.read_csv(x_path, header=None).values, dtype=torch.float64, device=device))
            Y_groups.append(torch.tensor(pd.read_csv(y_path, header=None).values, dtype=torch.float64, device=device))
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
pass

