# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/archived_code/33_DirectContestwROut.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # We use the 300 csv files created and implement a direct contest against the r codes.
# # Takes 14.8 Minutes per Simulation.

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
# # Global parameters:
number_of_cycles = 500 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
locations_per_group = 200 # how many locations to observe per group
number_of_locations = number_of_groups * locations_per_group # total locations
number_of_simulations = 300 # for synthetic data, how many optimisation to average over
steps_per_batch = 5
dims = 2  # 2D spatial
p =  3 # how many features 

# %%
X,Y,true_K, alpha_matrix_true, nu_matrix_true, sigma_matrix_true = Genton_parametrisation(number_of_locations, dims)
ground_truth_df = store_as_df(alpha_matrix_true, nu_matrix_true, sigma_matrix_true)
estimated_params_df = pd.DataFrame()
distance_K_df = pd.DataFrame()

for _ in range(1,301):
    print(f"data set {_}")
    try:
        # Load the CSV file
        df = pd.read_csv(f'~/project/synthetic_data/realisation_{_}.csv')
        # Extract the spatial coordinates (first and second spatial coordinates)
        X = torch.tensor(df[['0','1']].values[:200], dtype=torch.float64).detach()
        # Prepare Y by reshaping the gene expression levels for each gene
        expression = df['Expression'].values.reshape(200, 3)
        Y = torch.tensor(expression, dtype=torch.float64).detach()
        torch.autograd.set_detect_anomaly(True)
        optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
        alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
        estimated_K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
        distance_K = torch.norm(true_K - estimated_K)**2 / torch.norm(true_K)**2
        distance_K_df = pd.concat([distance_K_df, pd.DataFrame([distance_K.item()])] ,ignore_index=True)
        estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix) ], ignore_index=True)
        # Save the combined DataFrame to a CSV file
        file_path = os.path.expanduser(f'~/project/python_processed_data/synthetic_estimated_parameters_2.csv')
        estimated_params_df.to_csv(file_path, index=False)
    except Exception as e:
        print(e)
        continue
estimated_params_df

# %%
histograms_are_plotted = True
df_to_plot = estimated_params_df
%run -i epilogue

# %%
plt.hist(distance_K_df[0])

