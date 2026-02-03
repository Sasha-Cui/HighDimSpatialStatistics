# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/archived_code/39_gpu_61733_10.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # GPU: are we really running on GPU?

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
%load_ext memory_profiler

# %%
# # Global parameters:
number_of_cycles = 300 # how many passes through the training data we go through
number_of_groups = 3 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 1
dims = 2  # 2D spatial
head = 3000 # how many locations to consider in the real data set.

# %%
puck_list = ['Puck_230517_39'] # the largest puck
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
# gene_list = gene_list[:6]  # First 6 elements
# gene_list = gene_list[6:12]  # Next 6 elements
# gene_list = gene_list[12:18]  # Next 6 elements
gene_list = gene_list[18:]  # Remaining elements
adata, X,Y, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
# # Fitting Codes

# %%
%%memit
start_time = time.time()
optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:.6f} hours")

# %%
# Run this block only on slurm
df_to_plot = pd.DataFrame(optimized_marginal_params)
notebook_name = "39_1"
histograms_are_plotted = False
%run -i epilogue

# %%
# # The outputs are (alpha_i.item(), nu_i.item(), sigma_i.item()) of the genes.

# %%
optimized_marginal_params

# %%
gene_list

