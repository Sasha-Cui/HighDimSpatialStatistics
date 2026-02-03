# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/42_random_subsampler.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Goal of this notebook is subsample the locations randomly.
#
# 1. Input is just a directory that stores the data
# 2. Randomly pick locations
# 4. For each dataset, create a new directory that stores various smoothed / subsampled data, namely the content of $X_{groups}, Y_{groups}$

# %%
%load_ext memory_profiler

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
input_dir_base = '~/project/41_1_train_data'
output_dir_base = '~/project/42_randomly_subsampled_synthetic_data'
# output_dir_base = '~/project/42_subsampled_hattie_data'

# %%
os.makedirs(os.path.expanduser(output_dir_base), exist_ok=True)

# %%
# Read in the train data
#
#     X = ~/project/41_1_train_data/X_train_{i}.csv
#
#     Y = ~/project/41_1_train_data/Y_train_{i}_{j}.csv 
#
# subsample these using the function
#
#     X_groups, Y_groups = kernel_smoothing(X, Y, number_of_grids=10, min_grid_count=100, max_grid_count=3000)
#
# and output the subsampled data, for k in range(number_of_grids)
#
#     ~/project/42_subsampled_hattie_data/{i}/X_subsampled_{k}.csv 
#     
#     ~/project/42_subsampled_hattie_data/{i}/Y_subsampled_{k}_{j}.csv 

# %%
# # by the way, in case you want to see the distribution of distances to Nearest Neighbour-
# # Summary statistics
# summary_stats = {
#     "mean": nearest_distances.mean(),
#     "std": nearest_distances.std(),
#     "min": nearest_distances.min(),
#     "max": nearest_distances.max(),
#     "25th_percentile": np.percentile(nearest_distances, 25),
#     "50th_percentile": median,
#     "75th_percentile": np.percentile(nearest_distances, 75)
# }

# # Print summary statistics
# print("Summary Statistics for Nearest Neighbor Distances:")
# for key, value in summary_stats.items():
#     print(f"{key}: {value}")

# # Plot histogram of nearest neighbor distances
# plt.hist(nearest_distances, bins=80, edgecolor="black")
# plt.xlabel("Nearest Neighbor Distance")
# plt.ylabel("Frequency")
# plt.title("Histogram of Nearest Neighbor Distances")
# plt.show()

# %%
os.path.exists(os.path.expanduser(f"{input_dir_base}/X_train_{2}.csv"))

# %%
def random_subsampling(X, Y, num_of_groups=10, num_of_rows_per_group=300):
    """
    Randomly shuffle and divide X and Y into a specified number of groups with a given number of rows each.

    Parameters:
        X (torch.Tensor): The input feature tensor of shape (n_rows, n_features).
        Y (torch.Tensor): The target tensor of shape (n_rows, n_targets).
        num_of_groups (int): The number of groups to divide the data into.
        num_of_rows_per_group (int): The number of rows in each group.

    Returns:
        (list, list): Two lists of tensors (X_groups, Y_groups), where each list contains
                      the subsets of X and Y respectively.
    """
    # Ensure both X and Y have the same number of rows
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of rows."
    
    total_rows = num_of_groups * num_of_rows_per_group
    assert X.shape[0] >= total_rows, f"Insufficient rows: Require at least {total_rows} rows."
    
    # Concatenate X and Y along the columns for consistent shuffling
    combined = torch.cat((X, Y), dim=1)
    
    # Shuffle the combined data
    shuffled = combined[torch.randperm(combined.size(0))]
    
    # Split the shuffled data back into X and Y
    num_features_X = X.size(1)  # Number of features in X
    X_shuffled = shuffled[:, :num_features_X]
    Y_shuffled = shuffled[:, num_features_X:]
    
    # Divide the data into groups
    X_groups = [X_shuffled[i:i + num_of_rows_per_group] for i in range(0, total_rows, num_of_rows_per_group)]
    Y_groups = [Y_shuffled[i:i + num_of_rows_per_group] for i in range(0, total_rows, num_of_rows_per_group)]
    
    return X_groups, Y_groups

# %%
# # Example tensors
# X = torch.rand(3000, 5)  # 3000 rows, 5 features
# Y = torch.rand(3000, 2)  # 3000 rows, 2 targets

# # Apply the function
# X_groups, Y_groups = random_subsampling(X, Y)

# # Check results
# print(f"Number of groups: {len(X_groups)}")
# print(f"Rows per group in X: {X_groups[0].shape}")
# print(f"Rows per group in Y: {Y_groups[0].shape}")

# %%
# Iterate over the training data for different values of i
i = 0
while True:
    # Read the train data
    X_path = f"{input_dir_base}/X_train_{i}.csv"
    if not os.path.exists(os.path.expanduser(X_path)):
        print("finished at data set", i-1)
        break

    # code for random subsampling 
    X = pd.read_csv(os.path.expanduser(X_path), header=None)    
    X = torch.tensor(X.values, dtype=torch.float64)
    
    # Read all Y_train_{i}_j files
    Y_list = []
    j = 0
    while True:
        Y_path = f"{input_dir_base}/Y_train_{i}_{j}.csv"
        if not os.path.exists(os.path.expanduser(Y_path)):
            break
        Y_list.append(pd.read_csv(os.path.expanduser(Y_path), header=None))
        j += 1
    
    if not Y_list:
        break

    # Concatenate all Y files
    Y = pd.concat(Y_list, axis=1)
    Y = torch.tensor(Y.values, dtype=torch.float64)

    # Subsample
    X_groups, Y_groups = random_subsampling(X, Y, num_of_groups=10, num_of_rows_per_group=300)
    
    
    # Output the subsampled data
    output_dir = f"{output_dir_base}/{i}"
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    for k in range(len(X_groups)):
        X_output_path = f"{output_dir}/X_subsampled_{k}.csv"
        Y_output_path = f"{output_dir}/Y_subsampled_{k}.csv"
        
        pd.DataFrame(X_groups[k].numpy()).to_csv(os.path.expanduser(X_output_path), index=False, header=False)
        pd.DataFrame(Y_groups[k].numpy()).to_csv(os.path.expanduser(Y_output_path), index=False, header=False)
    
    i += 1

# %%
len(X_groups)

# %%
# sanity_checking_kernel_smoothing(X_groups, Y_groups, X, Y)

