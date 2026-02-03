# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/44_metric_calculation.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# <span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [4]</a>'.</span>

# %%
# # calculates the val metric after fitting is completed

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
# Directories
input_dir_base = '~/project/42_subsampled_synthetic_data'
output_dir_base = '~/project/43_estimation_results'

# %%
# ## Computing the validation metric

# %%
# The function validation_metric() does the following.  Based on the matern parameters computed for a single feature (i.e. optimized_marginal_params, which contains the estimated values for alpha, nu, and sigma), it constructs the covariance matrix K on a set of locations X_test using the function 
#
#     K_pred = approx_matern_kernel_marginal(X_test, alpha_i, nu_i, sigma_i)
#
# and then calculates the relative squared frobenius distance bewteen K_pred and K_test, namely (norm(K_pred-K_test)**2)/(norm(K_test)**2).

# %%
# def validation_metric(optimized_marginal_params_of_a_feature, X_test, K_test):
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
#             result = validation_metric(optimized_marginal_params, X_test, K_test)
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
#         result = validation_metric(optimized_marginal_params, X_test, K_test)
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
#                 result = validation_metric(optimized_marginal_params, X_test, K_test)
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
# # testing_for_validation_metric()

# %%
base_dir = os.path.expanduser('~/project')
input_X_test_dir = os.path.join(base_dir, '41_2_test_locations')
input_K_test_dir = os.path.join(base_dir, '41_3_test_cov')
optimized_params_dir = os.path.expanduser(output_dir_base)
output_validation_dir = os.path.expanduser(output_dir_base)

l = 5 # Hyperparam index
i = 0 # Dataset index
j = 0  # Feature index
# Load X_test and K_test
X_test_path = f"{input_X_test_dir}/X_test_{i}.csv"
X_test = torch.tensor(pd.read_csv(X_test_path, header=None).values, dtype=torch.float64)
K_test_path = f"{input_K_test_dir}/K_test_{i}_{j}.csv"
K_test = torch.tensor(pd.read_csv(K_test_path, header=None).values, dtype=torch.float64)
print("alpha,    nu,    alpha^(2*nu),   val_metric")
for alpha_init in [0.001, 0.01,0.1,1,10,100]:
    for nu_init in [0.0012, 0.012,0.12,1.2,12]:
        optimized_marginal_params = {
            'alpha': torch.tensor(alpha_init, dtype=torch.float64, device=device),
            'nu': torch.tensor(nu_init, dtype=torch.float64, device=device),
            'sigma': torch.tensor(1, dtype=torch.float64, device=device)
        }
        print(f"{alpha_init};    {nu_init};     {alpha_init**(2*nu_init):.4f};       {validation_metric(optimized_marginal_params, X_test, K_test):.8f}")

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
# <span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>

# %%
## Compute the validation metric on each feature j on each data set i
# Define directories
base_dir = os.path.expanduser('~/project')
input_X_test_dir = os.path.join(base_dir, '41_2_test_locations')
input_K_test_dir = os.path.join(base_dir, '41_3_test_cov')
optimized_params_dir = os.path.expanduser(output_dir_base)
output_validation_dir = os.path.expanduser(output_dir_base)

# Iterate over each hyperparameter setting `l`
# for l in range(len(os.listdir(optimized_params_dir))):
for l in range(0,9):
    i = 0  # Dataset index
    while True:
        # Define paths for X_test for the i-th dataset
        X_test_path = f"{input_X_test_dir}/X_test_{i}.csv"
        if not os.path.exists(X_test_path):
            print(f"collected up to data set {i-1} of hyperparamter {l}.")
            break  # Exit loop if X_test file doesn't exist for dataset `i`

        # Load X_test
        X_test = torch.tensor(pd.read_csv(X_test_path, header=None).values, dtype=torch.float64)

        j = 0  # Feature index
        while True:
            # Define paths for K_test and optimized parameters for feature `j`
            K_test_path = f"{input_K_test_dir}/K_test_{i}_{j}.csv"
            optimized_params_path = f"{optimized_params_dir}/{l}/best_parameters_dataset_{i}_feature_{j}.csv"
            if not os.path.exists(K_test_path) or not os.path.exists(optimized_params_path):
                break  # Exit loop if K_test or optimized parameters file doesn't exist for feature `j`

            # Load K_test and optimized parameters
            K_test = torch.tensor(pd.read_csv(K_test_path, header=None).values, dtype=torch.float64)
            optimized_params_df = pd.read_csv(optimized_params_path)
            optimized_params = {
                'alpha': torch.tensor(optimized_params_df['alpha'].values[0], dtype=torch.float64),
                'nu': torch.tensor(optimized_params_df['nu'].values[0], dtype=torch.float64),
                'sigma': torch.tensor(optimized_params_df['sigma'].values[0], dtype=torch.float64)
            }

            # Compute validation metric using the existing validation_metric function
            validation_metric_value = validation_metric(optimized_params, X_test, K_test)

            # Append the metric for the dataset `i`
            validation_metrics = {'data set i': i, 'validation_metric': validation_metric_value}
            
            # Append metric to the DataFrame for each feature `j`
            output_validation_path = f"{output_validation_dir}/validation_metric_hyperparam_{l}_feature_{j}.csv"
            if os.path.exists(output_validation_path):
                validation_df = pd.read_csv(output_validation_path)
                validation_df = pd.concat([validation_df, pd.DataFrame([validation_metrics])], ignore_index=True)
            else:
                validation_df = pd.DataFrame([validation_metrics])
            # Save the updated validation metrics DataFrame
            validation_df.to_csv(output_validation_path, index=False)
            
            j += 1  # Move to the next feature `j`

        i += 1  # Move to the next dataset `i`

print("Per feature, per data set validation metrics computed and stored.")

# %%
## Compute the averages on each feature j, across different data sets i
# Loop through all files in the directory that match the feature file pattern
for filename in os.listdir(output_validation_dir):
    if filename.startswith("validation_metric_hyperparam_") and filename.endswith(".csv"):
        # Construct full path for the file
        feature_validation_path = os.path.expanduser(f"{output_validation_dir}/{filename}")

        # Read the CSV file
        validation_df = pd.read_csv(feature_validation_path)

        # data_types = validation_df["data set i"].apply(type).value_counts()
        # print("Data types in the column 'data set i':")
        # print(data_types)

        # Skip files without the expected "validation_metric" column
        if validation_df.columns[1]!="validation_metric":
            print(f"'validation_metric' row does not exists in {filename}. Skipping this file.")
            continue

        # Check if an "average" row already exists
        if validation_df["data set i"].astype(str).str.strip().eq("average").any():
            print(f"'average' row already exists in {filename}. Skipping this file.")
            continue
        # Calculate the average validation metric
        avg_metric = validation_df["validation_metric"].mean()
        
        # Create an average row and append it to the DataFrame
        avg_row = pd.DataFrame([{"data set i": "average", "validation_metric": avg_metric}])
        validation_df = pd.concat([validation_df, avg_row], ignore_index=True)
        
        # Save the updated DataFrame back to the same file
        validation_df.to_csv(feature_validation_path, index=False)
        print(f"Updated {filename} with an average row.")

print("Per feature validation metrics averaged over data sets computed and stored.")

# %%
## Compute the averages across different data sets i and different features
import pandas as pd
import glob
import os
import statistics


# Define base directories and expand the user path
input_dir_base = os.path.expanduser('~/project/42_subsampled_synthetic_data')
output_dir_base = output_dir_base

# Initialize l and start the loop
l = 0
while True:
    # Define the file pattern for the current value of l with expanded output_dir_base
    file_pattern = os.path.expanduser(f"{output_dir_base}/validation_metric_hyperparam_{l}_feature_*.csv")
    # print(f"Looking for files with pattern: {file_pattern}")  # Debug print
    
    files = glob.glob(file_pattern)
    
    # Check if any files are found and print them for debugging
    if not files:
        print(f"No files found for hyperparameter {l}. Ending loop.")  # Debug print
        break
    
    # Initialize a list to store the averages across features
    feature_averages = []
    
    # Loop through each file and extract the average value from the "average" row
    for file in files:
        # print(f"Processing file: {file}")  # Debug print
        df = pd.read_csv(file, delimiter=',')  # Adjust delimiter if needed
        
        # Strip any leading/trailing whitespace from column headers
        df.columns = df.columns.str.strip()
        # print("Columns in the file after stripping whitespace:", df.columns)  # Debug column names
        
        # Ensure 'data set i' is in the columns after stripping
        if 'data set i' not in df.columns:
            raise KeyError(f"Column 'data set i' not found in {file}. Available columns: {df.columns}")

        
        # Extract the average value from the "average" row
        feature_average = df.loc[df['data set i'] == 'average', 'validation_metric'].values[0]
        feature_averages.append(feature_average)
    
    # Compute the overall average across features
    overall_average = statistics.mean(feature_averages) if feature_averages else None
    overall_median = statistics.median(feature_averages) if feature_averages else None
    
    # Store the result in a new DataFrame and save to a CSV if we have valid data
    if overall_average is not None:
        output_filename = os.path.expanduser(f"{output_dir_base}/validation_metric_hyperparam_{l}.csv")
        output_df = pd.DataFrame([
        ["hyperparameter", int(l)],
        ["average_validation_metric", overall_average],
        ["median_validation_metric", overall_median]
        ])
        output_df.to_csv(output_filename, index=False, header=False)
        print(f"Saved metric mean and median to {output_filename}")  # Debug print
    
    # Increment l for the next iteration
    l += 1
    
print("Validation metrics averaged over data sets and features computed and stored.")

# %%
pass

