# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/37_MemoryTracking.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # This notebook tracks the amount of memory needed to do the optimisation.  Note that we only run for 1 step each cycle.  Otherwise the time taken will be too long. 

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
# # Global parameters:
number_of_cycles = 1 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 1
dims = 2  # 2D spatial
head = 0 # how many locations to consider in the real data set.

# %%
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
# gene_list = ["Inha", "Esr1"]
# puck_list = ['Puck_230223_01', 'Puck_230406_01', 'Puck_230406_06', 'Puck_230406_08', 
#              'Puck_230517_37', 'Puck_230517_38', 'Puck_230517_39', 'Puck_230913_07', 
#              'Puck_240108_20', 'Puck_240108_24', 'Puck_240108_25', 'Puck_240108_26', 
#              'A0029_047', 'A0029_043', 'Puck_230807_27', 'Puck_240129_36', 'Puck_240129_37', 
#              'A0029_042', 'A0029_036', 'PM104_004', 'Puck_240108_10', 'Puck_240108_11', 
#              'Puck_230807_04', 'Puck_230714_28', 'Puck_230714_23']
puck_list = 'all'

# puck_list = ['Puck_230517_39'] # the largest puck
adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
# ### The memory tracker

# %%
# this block runs the loading and the optimisation steps 
def fitting_codes(head):
    adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)
    estimated_params_df = pd.DataFrame()
    distance_K_df = pd.DataFrame()
    torch.autograd.set_detect_anomaly(True)
    optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
    # print(optimized_marginal_params)
    alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
    # estimated_K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    # distance_K = torch.norm(true_K - estimated_K)**2 / torch.norm(true_K)**2
    # distance_K_df = pd.concat([distance_K_df, pd.DataFrame([distance_K.item()])] ,ignore_index=True)
    estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix) ], ignore_index=True)

    return estimated_params_df

# %%
# Memory & Time tracking function
# Load the memory profiler magic
%load_ext memory_profiler
import time
def main(head):
    start_time = time.time()
    fitting_codes(head)
    end_time = time.time()
    elapsed_time =(end_time - start_time)/3600
    print(f"Time taken: {elapsed_time:.6f} hours")
    return True

# %%
%memit main(5500)

# %%
# ### Memory Profiling ends

# %%
# ### At this stage, we have obtained our data.

# %%
# estimated_params_df = pd.DataFrame()
# distance_K_df = pd.DataFrame()
# for _ in range(1):
#     try:
#         torch.autograd.set_detect_anomaly(True)
#         optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
#         print(optimized_marginal_params)
#         alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
#         # estimated_K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
#         # distance_K = torch.norm(true_K - estimated_K)**2 / torch.norm(true_K)**2
#         # distance_K_df = pd.concat([distance_K_df, pd.DataFrame([distance_K.item()])] ,ignore_index=True)
#         estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix) ], ignore_index=True)
#     except Exception as e:
#         print(e)
#         continue
# estimated_params_df

# %%
# estimated_params_df = pd.DataFrame()
# distance_K_df = pd.DataFrame()
# for _ in range(number_of_simulations):
#     try:
#         optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
#         alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
#         estimated_K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
#         estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix) ], ignore_index=True)
#     except Exception as e:
#         print(e)
#         continue
# estimated_params_df

# %%
# import matplotlib.pyplot as plt
# import math

# # Get the list of columns
# columns = estimated_params_df.columns

# # Calculate the number of rows and columns for subplots
# n_params = len(columns)
# n_cols = 3  # Fixed number of columns for layout
# n_rows = math.ceil(n_params / n_cols)  # Calculate required rows based on the number of parameters

# plt.figure(figsize=(15, 2.5 * n_rows))  # Adjust height based on number of rows
# for i, col in enumerate(columns):
#     plt.subplot(n_rows, n_cols, i + 1)
    
#     # Plot the histogram of estimates
#     plt.hist(estimated_params_df[col], bins=30, color='skyblue', edgecolor='black')
    
#     # Plot the vertical line for the true value
#     # plt.axvline(x=ground_truth_df[col].iloc[0], color='red', linestyle='--', linewidth=2)
    
#     plt.title(f'{col} Distribution')
#     plt.xlabel(f'{col}')
#     plt.ylabel('Frequency')
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

# %%
# Save the computed parameters

# %%
# # Concatenate the two DataFrames along the columns (axis=1)
# # combined_df = pd.concat([estimated_params_df, distance_K_df], axis=1)
# # # Define the file path using f-string and expand the home directory
# file_path = os.path.expanduser(f'~/project/python_processed_data/estimated_parameters_4.csv')
# # # Save the combined DataFrame to a CSV file
# # combined_df.to_csv(file_path, index=False)
# # combined_df
# estimated_params_df.to_csv(file_path, index=False)

