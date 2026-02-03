# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/54_finding_best_param_real_data.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # for each feature $j$, we look through hyperparams to find $l$ the smallest minimal loss.  We report both best param, optimized param, as well as the loss history for that $l$

# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the base directory and expand the '~' to the home directory
base_dir = os.path.expanduser('~/project/53_hattie_marginal_estimation_results')

# Get all hyperparameter files
hyperparam_files = sorted(glob.glob(os.path.join(base_dir, 'hyperparameters_*.csv')))

# Extract all unique feature indices across all directories
all_feature_indices = set()
for hyperparam_file in hyperparam_files:
    # Extract 'l' from the filename
    l = os.path.splitext(os.path.basename(hyperparam_file))[0].split('_')[-1]
    l = int(l)

    # Directory containing loss histories
    loss_history_dir = os.path.join(base_dir, str(l))
    loss_files = glob.glob(os.path.join(loss_history_dir, 'loss_histories_dataset_0_feature_*.csv'))
    feature_indices = [
        int(os.path.splitext(os.path.basename(f))[0].split('_')[-1])
        for f in loss_files
    ]
    all_feature_indices.update(feature_indices)

# Dictionary to store the best results for each feature
best_results = {}

# Process each feature index
for j in sorted(all_feature_indices):
    best_result_for_j = None

    # Scan through all choices of `l`
    for hyperparam_file in hyperparam_files:
        # Extract 'l' from the filename
        l = os.path.splitext(os.path.basename(hyperparam_file))[0].split('_')[-1]
        l = int(l)

        # Directory containing data for this value of l
        data_dir = os.path.join(base_dir, str(l))

        # Paths to the necessary files for feature j
        loss_history_path = os.path.join(data_dir, f'loss_histories_dataset_0_feature_{j}.csv')
        best_params_path = os.path.join(data_dir, f'best_parameters_dataset_0_feature_{j}.csv')
        optimized_params_path = os.path.join(data_dir, f'optimized_parameters_dataset_0_feature_{j}.csv')

        # Check if all necessary files exist
        if all(os.path.exists(path) for path in [loss_history_path, best_params_path, optimized_params_path]):
            # Read the loss history for feature j
            loss_history_df = pd.read_csv(loss_history_path)
            loss_values = loss_history_df['loss'].values  # Assuming 'loss' is a column name

            # Find the smallest minimal loss for this `l`
            min_loss = np.min(loss_values)

            # Read the best and optimized parameters
            best_params_df = pd.read_csv(best_params_path)
            best_params = best_params_df.to_dict(orient='records')[0]  # Extract the row as a dictionary
            
            optimized_params_df = pd.read_csv(optimized_params_path)
            optimized_params = optimized_params_df.to_dict(orient='records')[0]  # Extract the row as a dictionary

            # Check if this result is the best so far for feature `j`
            if best_result_for_j is None or min_loss < best_result_for_j['min_loss']:
                best_result_for_j = {
                    'l': l,
                    'feature_index': j,
                    'min_loss': min_loss,
                    'best_params': best_params,
                    'optimized_params': optimized_params,
                    'loss_history': loss_values.tolist()
                }

    # Store the best result for this feature
    if best_result_for_j:
        best_results[j] = best_result_for_j

# Convert the results into a DataFrame
results_df = pd.DataFrame.from_dict(best_results, orient='index')
results_df

# %%
# Save the results to a CSV file
results_output_path = os.path.join(base_dir, 'best_results_per_feature.csv')
results_df.to_csv(results_output_path, index=False)

print(f"Results saved to {results_output_path}")

# %%
import matplotlib.pyplot as plt

# Plot each loss history from the DataFrame
for feature_index, loss_history in enumerate(results_df["loss_history"]):
    plt.figure(figsize=(8, 4))  # Create a new figure for each feature
    plt.plot(loss_history, label=f'Feature {feature_index}', linewidth=2)
    plt.title(f'Min Loss History for Feature {feature_index}', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
pass

