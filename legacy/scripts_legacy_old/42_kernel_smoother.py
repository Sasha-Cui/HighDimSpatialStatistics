# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/42_kernel_smoother.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Goal of this notebook is to read in locations, kernel smooth them into various grids.
#
# 1. Input is just a directory that stores the data
# 2. Create grids of various sizes
# 3. Kernel Smooth
# 4. For each dataset, create a new directory that stores various smoothed / subsampled data, namely the content of $X_{groups}, Y_{groups}$

# %%
%load_ext memory_profiler

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
input_dir_base = '~/project/41_1_train_data'
output_dir_base = '~/project/42_subsampled_synthetic_data'
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
from scipy.spatial.distance import pdist, squareform
# Iterate over the training data for different values of i
i = 0
while True:
    # Read the train data
    X_path = f"{input_dir_base}/X_train_{i}.csv"
    if not os.path.exists(os.path.expanduser(X_path)):
        print("finished at data set", i-1)
        break

    
    # code to compute the median and therefore the bandwidth
    X = pd.read_csv(os.path.expanduser(X_path), header=None)    
    pairwise_distances = pdist(X)  # Compute pairwise distances using pdist
    distance_matrix = squareform(pairwise_distances)  # Convert to a square matrix format
    
    # Set the diagonal to infinity to ignore self-distances
    np.fill_diagonal(distance_matrix, np.inf)
    
    # Find the minimum (nearest neighbor) distance for each point
    nearest_distances = distance_matrix.min(axis=1)
    median = np.percentile(nearest_distances, 50)
    bandwidth = median *2
    
    # code for kernel smoothing
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

    # Subsample using kernel smoothing
    X_groups, Y_groups = kernel_smoothing(X, Y, bandwidth, number_of_grids=10, min_grid_count=100, max_grid_count=4000)
    
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

