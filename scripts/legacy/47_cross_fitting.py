# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/47_cross_fitting.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Takes fitted marginal parameters and performs cross terms fitting. 
# 1. As before, tries different hyperparameters.
# 2. Additionally we are storing both the Genton parameters (with $\rho, \Delta_A, \Delta_B, W$) and the Matern parameters ($\alpha, \nu, \sigma$).
#
# Note, the Genton parameters are converted to the Matern parameters through
#     
#     alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)

# %%
# We run through different hyperparameters through 
#
#     for l, lr_set in enumerate(lr_combinations):
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
#     '~/project/44_cross_estimation_results/hyperparameters_{l}.csv'
#     
# The nominal outputs are the estimated parameters obtained by the function
#
#     alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
#
#     optimized_alpha_matrix, optimized_nu_matrix, optimized_sigma_matrix, best_alpha_matrix, best_nu_matrix, best_sigma_matrix, loss_histories = optimize_cross_parameters_in_groups(lr_set, init_set, X_groups, Y_groups, number_of_cycles, steps_per_batch, print_early_stopping_epochs = True)
#     
# and the estimation results are then stored in
#     
#     '~/project/44_cross_estimation_results/{l}/optimized_parameters_dataset_{i}.csv'
#     '~/project/44_cross_estimation_results/{l}/best_parameters_dataset_{i}.csv'
#
# We also store loss_histories over the training process in
#     
#     '~/project/44_cross_estimation_results/{l}/loss_histories_dataset_{i}/'
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

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
# ## First half, computing the estimated parameters for each hyperparameter setting.

# %%
from itertools import product
# Define possible base learning rates and scaling factors for each parameter
base_lr_options = {
    'Delta_A': [1e-4],
    'Delta_B': [1e-4],
    'rho_A': [1e-4],
    'rho_B': [1e-4],
    'rho_V': [1e-4],
    'W': [1e-4]
}

scaling_factor_options = {
    'Delta_A': [10, 100],
    'Delta_B': [10, 100],
    'rho_A': [10, 100],
    'rho_B': [10, 100],
    'rho_V': [10, 100],
    'W': [10, 100]
}
# Generate all combinations of base learning rates and scaling factors
lr_combinations = []
for base_lrs, scaling_factors in product(
        product(*base_lr_options.values()), product(*scaling_factor_options.values())):
    lr_set = {
        name: base_lr * scaling
        for name, base_lr, scaling in zip(base_lr_options.keys(), base_lrs, scaling_factors)
    }
    lr_combinations.append(lr_set)

# %%
number_of_cycles, steps_per_batch = 10 , 1
input_dir_base = '~/project/42_subsampled_synthetic_data'
output_dir_base = '~/project/44_cross_estimation_results'

# %%
# Define the values for alpha, nu, and sigma
alpha_values = [0.01, 0.02, 0.03]
nu_values = [1.2, 0.6, 0.3]
sigma_values = [1, 1, 1]
optimized_marginal_params = [
    torch.tensor([alpha, nu, sigma], dtype=torch.float64)
    for alpha, nu, sigma in zip(alpha_values, nu_values, sigma_values)
]

print("optimized_marginal_params:", optimized_marginal_params)

# %%
for l, lr_set in enumerate(lr_combinations):
    # print(f"Combination {l}: {lr_set}")
    print("hyperparameter ", l)
    # Define output directories for the current hyperparameter setting `l`
    optimized_dir = os.path.expanduser(f'{output_dir_base}/{l}')
    os.makedirs(optimized_dir, exist_ok=True)
    i = 0
    while os.path.exists(os.path.expanduser(f'{input_dir_base}/{i}')):
        X_groups, Y_groups = [], []
        k = 0
        # Load all X_ and Y_groups (k) of data for each dataset (i)
        while True:
            x_path = os.path.expanduser(f'{input_dir_base}/{i}/X_subsampled_{k}.csv')
            y_path = os.path.expanduser(f'{input_dir_base}/{i}/Y_subsampled_{k}.csv')
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                break
            X_groups.append(torch.tensor(pd.read_csv(x_path).values, dtype=torch.float64))
            Y_groups.append(torch.tensor(pd.read_csv(y_path).values, dtype=torch.float64))
            k += 1

        # Store hyperparameters
        hyperparameters_path = os.path.expanduser(f'{output_dir_base}/hyperparameters_{l}.csv')
        pd.DataFrame([lr_set]).to_csv(hyperparameters_path, index=False)

        # Optimize parameters
        optimized_params, best_params, loss_histories = optimize_cross_parameters_in_groups(
            optimized_marginal_params, lr_set, X_groups, Y_groups, number_of_cycles=number_of_cycles,
            steps_per_batch=steps_per_batch, print_early_stopping_epochs=False, checkpoint_interval=50,
            max_time_hours=24, logging=True)
        
        # Collect all optimized parameters for this dataset `i`
        all_optimized_params = []  # List to hold all parameters for the current dataset `i`


        print(optimized_params)
        
        # Convert tensors to float values and prepare for storage
        def convert_params_to_floats(params):
            if isinstance(params, dict):
                return {
                    name: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v
                    for name, v in params.items()
                }
            elif isinstance(params, list):
                return [
                    v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v
                    for v in params
                ]
            else:
                raise TypeError("Unsupported type for params. Expected dict or list.")

        optimized_params = convert_params_to_floats(optimized_params)
        best_params = convert_params_to_floats(best_params)

        # Append parameters to `all_optimized_params`
        all_optimized_params.append({
            'dataset': i,
            'Delta_A_optimized': optimized_params['Delta_A'],
            'Delta_B_optimized': optimized_params['Delta_B'],
            'rho_A_optimized': optimized_params['rho_A'],
            'rho_B_optimized': optimized_params['rho_B'],
            'rho_V_optimized': optimized_params['rho_V'],
            'W_optimized': optimized_params['W'],
            'Delta_A_best': best_params['Delta_A'],
            'Delta_B_best': best_params['Delta_B'],
            'rho_A_best': best_params['rho_A'],
            'rho_B_best': best_params['rho_B'],
            'rho_V_best': best_params['rho_V'],
            'W_best': best_params['W']
        })

        # Store all optimized parameters for dataset `i` in a single CSV file
        optimized_params_path = f'{optimized_dir}/optimized_parameters_dataset_{i}.csv'
        pd.DataFrame(all_optimized_params).to_csv(optimized_params_path, index=False)

        i += 1  # Move to the next dataset index for the current hyperparameter setting
        # if i == 1:
            # break
print("Optimization process completed.")

# %%
optimized_params

# %%
# ## UNDER CONSTRUCTION Second half, computing the validation metric

# %%
# The function validation_metric() does the following.  Based on the matern parameters computed for a single feature (i.e. optimized_marginal_params, which contains the estimated values for alpha, nu, and sigma), it constructs the covariance matrix K on a set of locations X_test using the function 
#
#     K_pred = approx_matern_kernel_marginal(X_test, alpha_i, nu_i, sigma_i)
#
# and then calculates the relative squared frobenius distance bewteen K_pred and K_test, namely (norm(K_pred-K_test)**2)/(norm(K_test)**2).

# %%
# def validation_metric_marginal(optimized_marginal_params_of_a_feature, X_test, K_test):
#     """
#     Compute the validation metric as the relative squared Frobenius distance 
#     between the predicted and true covariance matrices.

#     Parameters:
#     - optimized_marginal_params (dict): Dictionary with estimated parameters (alpha, nu, sigma).
#     - X_test (torch.Tensor): Test locations tensor on the GPU.
#     - K_test (torch.Tensor): True covariance matrix tensor on the GPU.

#     Returns:
#     - validation_metric (float): Relative squared Frobenius distance between K_pred and K_test.
#     """
#     # Extract optimized parameters
#     alpha_j = optimized_marginal_params_of_a_feature['alpha']
#     nu_j = optimized_marginal_params_of_a_feature['nu']
#     sigma_j = optimized_marginal_params_of_a_feature['sigma']
    
#     # Get the device from X_test (assuming X_test and K_test are on the same device)
#     device = X_test.device
    
#     # Move parameters to the target device only if they are not already on it
#     if alpha_j.device != device:
#         alpha_j = alpha_j.to(device)
#     if nu_j.device != device:
#         nu_j = nu_j.to(device)
#     if sigma_j.device != device:
#         sigma_j = sigma_j.to(device)
    
#     # Compute the predicted covariance matrix K_pred using the Matérn kernel approximation
#     K_pred = approx_matern_kernel_marginal(X_test, alpha_j, nu_j, sigma_j)
    
#     # Compute the Frobenius norm squared of the difference
#     frobenius_diff = torch.norm(K_pred - K_test, p='fro') ** 2
    
#     # Compute the Frobenius norm squared of K_test
#     frobenius_K_test = torch.norm(K_test, p='fro') ** 2
    
#     # Calculate the relative squared Frobenius distance
#     validation_metric = frobenius_diff / frobenius_K_test
    
#     return validation_metric.item()
    
# def testing_for_validation_metric():
#     # Define simplified test cases
#     def test_basic_functionality():
#         """Test that validation_metric runs without error for typical inputs."""
#         X_test = torch.rand(10, 2, dtype=torch.float64, device=device)  # Set to float64 for consistency
#         K_test = torch.eye(10, dtype=torch.float64, device=device)  # Simple identity matrix for K_test in float64
#         optimized_marginal_params = {
#             'alpha': torch.tensor(1.0, dtype=torch.float64, device=device),
#             'nu': torch.tensor(0.5, dtype=torch.float64, device=device),
#             'sigma': torch.tensor(0.2, dtype=torch.float64, device=device)
#         }
        
#         try:
#             result = validation_metric_marginal(optimized_marginal_params, X_test, K_test)
#             print("Basic functionality test passed:", result)
#         except Exception as e:
#             print("Basic functionality test failed:", e)


#     def test_consistency_known_values():
#         """Test against known values for a simple setup where K_pred == K_test."""
#         X_test = torch.rand(100, 2, device=device)
#         # Define alpha, nu, and sigma as tensors to match approx_matern_kernel_marginal's requirements
#         alpha = torch.tensor(1.0, dtype=torch.float64, device=device)
#         nu = torch.tensor(0.5, dtype=torch.float64, device=device)
#         sigma = torch.tensor(0.2, dtype=torch.float64, device=device)
    
#         # Compute K_test with these parameters to make it match K_pred
#         K_test = approx_matern_kernel_marginal(X_test, alpha, nu, sigma)
    
#         # Define optimized_marginal_params with tensors
#         optimized_marginal_params = {'alpha': alpha, 'nu': nu, 'sigma': sigma}
    
#         # Calculate validation metric
#         result = validation_metric_marginal(optimized_marginal_params, X_test, K_test)
#         assert abs(result) < 1e-6, "Consistency test failed: Expected metric near 0"
#         print("Consistency test with known values passed:", result)


#     def test_edge_case_empty_input():
#         """Test edge case where X_test and K_test are empty tensors."""
#         X_test = torch.empty(0, 2, device=device)
#         K_test = torch.empty(0, 0, device=device)
#         optimized_marginal_params = {'alpha': torch.tensor(1.0, device=device),
#                                      'nu': torch.tensor(0.5, device=device),
#                                      'sigma': torch.tensor(0.2, device=device)}
#         try:
#             result = validation_metric(optimized_marginal_params, X_test, K_test)
#             print("Edge case (empty input) test passed:", result)
#         except Exception as e:
#             print("Edge case (empty input) test failed:", e)

#     def test_gpu_compatibility():
#         """Test GPU compatibility by running on GPU if available."""
#         if torch.cuda.is_available():
#             X_test = torch.rand(10, 2, device="cuda")
#             K_test = torch.eye(10, device="cuda")
#             optimized_marginal_params = {'alpha': torch.tensor(1.0, device="cuda"),
#                                          'nu': torch.tensor(0.5, device="cuda"),
#                                          'sigma': torch.tensor(0.2, device="cuda")}
#             try:
#                 result = validation_metric_marginal(optimized_marginal_params, X_test, K_test)
#                 print("GPU compatibility test passed:", result)
#             except Exception as e:
#                 print("GPU compatibility test failed:", e)
#         else:
#             print("GPU not available. Skipping GPU compatibility test.")

#     # Run all tests
#     print("Running tests for validation_metric...")
#     test_basic_functionality()
#     test_consistency_known_values()
#     test_edge_case_empty_input()
#     test_gpu_compatibility()
#     print("All tests completed.")

# # Example call to run the tests
# # testing_for_validation_metric_marginal()

# %%
# Having computed the estimated parameters, we can compute the validation metrics.  
#
# We need to load in a few things things.  First, X_test from 
#     
#     project/41_2_test_locations/X_test_{i}.csv
#
# ; Second, K_test from
#
#     project/41_3_test_cov/K_test_{i}_{j}.csv
#
# ; third, optimized_marginal_params_of_a_feature from 
#
#     project/43_estimation_results/{l}/optimized_parameters_{i}_{j}.csv
#
# For each data set i, for each feature j, we compute the validation metric through the function
#     
#     validation_metric = validation_metric(optimized_marginal_params_of_a_feature, X_test, K_test)
#     
# Finally, store different values of i as different rows in the file
#
#     '~/project/43_estimation_results/validation_metric_hyperparam_{l}_feature_{j}.csv'

# %%
# ## Compute the validation metric on each feature j on each data set i
# # Define directories
# base_dir = os.path.expanduser('~/project')
# input_X_test_dir = os.path.join(base_dir, '41_2_test_locations')
# input_K_test_dir = os.path.join(base_dir, '41_3_test_cov')
# optimized_params_dir = output_dir_base
# output_validation_dir = output_dir_base

# # Iterate over each hyperparameter setting `l`
# for l in range(len(os.listdir(optimized_params_dir))):
#     i = 0  # Dataset index
#     while True:
#         # Define paths for X_test for the i-th dataset
#         X_test_path = f"{input_X_test_dir}/X_test_{i}.csv"
#         if not os.path.exists(X_test_path):
#             break  # Exit loop if X_test file doesn't exist for dataset `i`

#         # Load X_test
#         X_test = torch.tensor(pd.read_csv(X_test_path).values, dtype=torch.float64)

#         j = 0  # Feature index
#         while True:
#             # Define paths for K_test and optimized parameters for feature `j`
#             K_test_path = f"{input_K_test_dir}/K_test_{i}_{j}.csv"
#             optimized_params_path = f"{optimized_params_dir}/{l}/best_parameters_dataset_{i}_feature_{j}.csv"
#             if not os.path.exists(K_test_path) or not os.path.exists(optimized_params_path):
#                 break  # Exit loop if K_test or optimized parameters file doesn't exist for feature `j`

#             # Load K_test and optimized parameters
#             K_test = torch.tensor(pd.read_csv(K_test_path).values, dtype=torch.float64)
#             optimized_params_df = pd.read_csv(optimized_params_path)
#             optimized_params = {
#                 'alpha': torch.tensor(optimized_params_df['alpha'].values[0], dtype=torch.float64),
#                 'nu': torch.tensor(optimized_params_df['nu'].values[0], dtype=torch.float64),
#                 'sigma': torch.tensor(optimized_params_df['sigma'].values[0], dtype=torch.float64)
#             }

#             # Compute validation metric using the existing validation_metric function
#             validation_metric_value = validation_metric(optimized_params, X_test, K_test)

#             # Append the metric for the dataset `i`
#             validation_metrics = {'data set i': i, 'validation_metric': validation_metric_value}
            
#             # Append metric to the DataFrame for each feature `j`
#             output_validation_path = f"{output_validation_dir}/validation_metric_hyperparam_{l}_feature_{j}.csv"
#             if os.path.exists(output_validation_path):
#                 validation_df = pd.read_csv(output_validation_path)
#                 validation_df = pd.concat([validation_df, pd.DataFrame([validation_metrics])], ignore_index=True)
#             else:
#                 validation_df = pd.DataFrame([validation_metrics])
#             # Save the updated validation metrics DataFrame
#             validation_df.to_csv(output_validation_path, index=False)
            
#             j += 1  # Move to the next feature `j`

#         i += 1  # Move to the next dataset `i`

# print("Per feature, per data set validation metrics computed and stored.")

# %%
# ## Compute the averages on each feature j, across different data sets i
# # Loop through all files in the directory that match the feature file pattern
# for filename in os.listdir(output_validation_dir):
#     if filename.startswith("validation_metric_hyperparam_") and filename.endswith(".csv"):
#         # Construct full path for the file
#         feature_validation_path = f"{output_validation_dir}/{filename}"

#         # Read the CSV file
#         validation_df = pd.read_csv(feature_validation_path)

#         # Calculate the average validation metric
#         avg_metric = validation_df["validation_metric"].mean()
        
#         # Create an average row and append it to the DataFrame
#         avg_row = pd.DataFrame([{"data set i": "average", "validation_metric": avg_metric}])
#         validation_df = pd.concat([validation_df, avg_row], ignore_index=True)
        
#         # Save the updated DataFrame back to the same file
#         validation_df.to_csv(feature_validation_path, index=False)

# print("Per feature validation metrics averaged over data sets computed and stored.")

# %%
# ## Compute the averages across different data sets i and different features
# import pandas as pd
# import glob
# import os


# # Define base directories and expand the user path
# input_dir_base = os.path.expanduser('~/project/42_subsampled_synthetic_data')
# output_dir_base = output_dir_base

# # Initialize l and start the loop
# l = 0
# while True:
#     # Define the file pattern for the current value of l with expanded output_dir_base
#     file_pattern = os.path.expanduser(f"{output_dir_base}/validation_metric_hyperparam_{l}_feature_*.csv")
#     # print(f"Looking for files with pattern: {file_pattern}")  # Debug print
    
#     files = glob.glob(file_pattern)
    
#     # Check if any files are found and print them for debugging
#     if not files:
#         print(f"No files found for hyperparameter {l}. Ending loop.")  # Debug print
#         break
    
#     # Initialize a list to store the averages across features
#     feature_averages = []
    
#     # Loop through each file and extract the average value from the "average" row
#     for file in files:
#         # print(f"Processing file: {file}")  # Debug print
#         df = pd.read_csv(file, delimiter=',')  # Adjust delimiter if needed
        
#         # Strip any leading/trailing whitespace from column headers
#         df.columns = df.columns.str.strip()
#         # print("Columns in the file after stripping whitespace:", df.columns)  # Debug column names
        
#         # Ensure 'data set i' is in the columns after stripping
#         if 'data set i' not in df.columns:
#             raise KeyError(f"Column 'data set i' not found in {file}. Available columns: {df.columns}")
        
#         # Extract the average value from the "average" row
#         feature_average = df.loc[df['data set i'] == 'average', 'validation_metric'].values[0]
#         feature_averages.append(feature_average)
    
#     # Compute the overall average across features
#     overall_average = sum(feature_averages) / len(feature_averages) if feature_averages else None
    
#     # Store the result in a new DataFrame and save to a CSV if we have valid data
#     if overall_average is not None:
#         output_filename = os.path.expanduser(f"{output_dir_base}/validation_metric_hyperparam_{l}.csv")
#         output_df = pd.DataFrame({
#             'hyperparameter': [l],
#             'average_validation_metric': [overall_average]
#         })
#         output_df.to_csv(output_filename, index=False)
#         print(f"Saved averaged metrics to {output_filename}")  # Debug print
    
#     # Increment l for the next iteration
#     l += 1
    
# print("Validation metrics averaged over data sets and features computed and stored.")

