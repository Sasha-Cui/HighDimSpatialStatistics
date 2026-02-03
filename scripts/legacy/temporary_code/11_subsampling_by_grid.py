# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/11_subsampling_by_grid.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Develops the subsampling by regular rectangular grid 

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
%load_ext memory_profiler
# # Global parameters:
number_of_cycles = 50 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 5
dims = 2  # 2D spatial
head = 5000 # how many locations to consider in the real data set.

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
X_groups, Y_groups = kernel_smoothing(X, Y, number_of_grids=10, min_grid_count=100, max_grid_count=3000)

# %%
X_groups

# %%
min_grid_count=100
max_grid_count=3000
torch.round(torch.sqrt(torch.linspace(min_grid_count, max_grid_count, steps=number_of_grids)))**2



# %%
pass

