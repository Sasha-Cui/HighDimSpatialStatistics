# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/40_CrossFitting.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Giving marginal fitting results, fit the cross terms

# %%
# ## Preambles

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
# Load the memory profiler magic
%load_ext memory_profiler

# %%
# ## Combine the CSVs from marginal fitting

# %%
# Next time, just write these as "combined_parameters_1"

# %%
# # run this only once!
# indices = [4, 3, 2, 1] # provide the correct indices in the correct order
# filepaths = [os.path.expanduser(f"~/project/python_processed_data/fitted_parameters_{i}.csv") for i in indices]
# df = pd.concat([pd.read_csv(filepath) for filepath in filepaths])
combined_df_index = 1
filepath = os.path.expanduser(f"~/project/python_processed_data/fitted_parameters_{combined_df_index}.csv")
# df.to_csv(filepath, index=False)
optimized_marginal_params = pd.read_csv(filepath).values.tolist()# df to list

# %%
optimized_marginal_params

# %%
# ## Cross Fitting Codes

# %%
# # Global parameters:
number_of_cycles = 50 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 5
dims = 2  # 2D spatial
head = 500 # how many locations to consider in the real data set.

# %%
puck_list = ['Puck_230517_39'] # the largest puck
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
# puck_list = ['Puck_230223_01', 'Puck_230406_01', 'Puck_230406_06', 'Puck_230406_08', 
#              'Puck_230517_37', 'Puck_230517_38', 'Puck_230517_39', 'Puck_230913_07', 
#              'Puck_240108_20', 'Puck_240108_24', 'Puck_240108_25', 'Puck_240108_26', 
#              'A0029_047', 'A0029_043', 'Puck_230807_27', 'Puck_240129_36', 'Puck_240129_37', 
#              'A0029_042', 'A0029_036', 'PM104_004', 'Puck_240108_10', 'Puck_240108_11', 
#              'Puck_230807_04', 'Puck_230714_28', 'Puck_230714_23']
# puck_list = 'all'
adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
estimated_params_df = pd.DataFrame()

# %%
%%memit 
start_time = time.time()
estimated_params_df = pd.DataFrame()
alpha_matrix, nu_matrix, sigma_matrix = optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles,steps_per_batch)
estimated_params_df = pd.concat([estimated_params_df, store_as_df(alpha_matrix, nu_matrix, sigma_matrix)], ignore_index=True)
elapsed_time =(time.time() - start_time)/3600
print(f"Time taken: {elapsed_time:.6f} hours")
estimated_params_df

# %%
# Run this block only on slurm
df_to_plot = pd.DataFrame(optimized_marginal_params)
notebook_name = "40"
histograms_are_plotted = False
%run -i epilogue

