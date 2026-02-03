# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/archived_code/36_SubsamplingwReplacementWeek.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # We subsample 20 times, with replacement, and check if the fitting results are comparable to each other

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
histograms_are_plotted = False

# %%
import time

# %%
# # Global parameters:
number_of_cycles = 500 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 5
dims = 2  # 2D spatial
head = 1000 # how many to down/subsample to.

# %%
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
# gene_list = [ "Inha"]
# puck_list = 'all'
puck_list = ['Puck_230517_39'] # the largest puck

adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)
df

# %%
start_time = time.time()
time.ctime(start_time)

# %%
head = 200 # how many locations to consider in the real data set.
estimated_params_df = pd.DataFrame()
for _ in range(20): # sub sampling with replacement the top head rows
    try:
        adata, X,Y,df=load_data(gene_list=gene_list, head=head, puck_list = puck_list) 
        torch.autograd.set_detect_anomaly(True)
        optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
        alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
        estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix) ], ignore_index=True)
    except Exception as e:
        print(e)
        continue
estimated_params_df

# %%
df_to_plot = estimated_params_df
%run -i epilogue
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_hours = elapsed_time_seconds / 3600
print(f"Time taken: {elapsed_time_hours:.6f} hours")

# %%
start_time = time.time()
time.ctime(start_time)

# %%
head = 500 # how many locations to consider in the real data set.
estimated_params_df = pd.DataFrame()
for _ in range(20): # sub sampling with replacement the top head rows
    try:
        adata, X,Y,df=load_data(gene_list=gene_list, head=head, puck_list = puck_list) 
        torch.autograd.set_detect_anomaly(True)
        optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
        alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
        estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix) ], ignore_index=True)
    except Exception as e:
        print(e)
        continue
estimated_params_df

# %%
df_to_plot = estimated_params_df
%run -i epilogue
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_hours = elapsed_time_seconds / 3600
print(f"Time taken: {elapsed_time_hours:.6f} hours")

# %%
start_time = time.time()
time.ctime(start_time)

# %%
head = 1000 # how many locations to consider in the real data set.
estimated_params_df = pd.DataFrame()
for _ in range(20): # sub sampling with replacement the top head rows
    try:
        adata, X,Y,df=load_data(gene_list=gene_list, head=head, puck_list = puck_list) 
        torch.autograd.set_detect_anomaly(True)
        optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
        alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
        estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix) ], ignore_index=True)
    except Exception as e:
        print(e)
        continue
estimated_params_df

# %%
df_to_plot = estimated_params_df
%run -i epilogue
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_hours = elapsed_time_seconds / 3600
print(f"Time taken: {elapsed_time_hours:.6f} hours")

