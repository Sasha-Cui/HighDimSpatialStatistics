# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/08_quantisation_of_dist-Copy1.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # To quantise the possible distances and speed up to the computations of K

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
%load_ext memory_profiler
# # Global parameters:
number_of_cycles = 500 # how many passes through the training data we go through
number_of_groups = 100 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 5
dims = 2  # 2D spatial
head=0
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]

# %%
# puck_list = ['Puck_230517_39'] # the largest puck
puck_list = ['Puck_230223_01', 'Puck_230406_01', 'Puck_230406_06', 'Puck_230406_08', 
             'Puck_230517_37', 'Puck_230517_38', 'Puck_230517_39', 'Puck_230913_07', 
             'Puck_240108_20', 'Puck_240108_24', 'Puck_240108_25', 'Puck_240108_26', 
             'A0029_047', 'A0029_043', 'Puck_230807_27', 'Puck_240129_36', 'Puck_240129_37', 
             'A0029_042', 'A0029_036', 'PM104_004', 'Puck_240108_10', 'Puck_240108_11', 
             'Puck_230807_04', 'Puck_230714_28', 'Puck_230714_23']
for puck in puck_list:
    adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = [puck])
    X_dist = torch.cdist(X,X)
    # Step 1: Create a boolean mask that identifies the off-diagonal elements
    off_diagonal_mask = ~torch.eye(X_dist.size(0), dtype=bool)
    
    # Step 2: Use the mask to extract only the off-diagonal elements
    off_diagonal_elements = X_dist[off_diagonal_mask]
    
    # Step 3: Flatten the off-diagonal elements
    off_diagonal = off_diagonal_elements.flatten()
    flat_matrix = torch.flatten(off_diagonal)
    df = pd.DataFrame(flat_matrix)

    # Step 4: Print Summary Statistics
    print(df.describe())
    print(df.max()/df.min())

# %%
pass

