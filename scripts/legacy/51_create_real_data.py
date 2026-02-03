# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/51_create_real_data.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Goal of this notebook is to put Hattie's data into the directory

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
puck_list = ['Puck_230517_39'] # the largest puck
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1",
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr",
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
head = 0
adata, X,Y, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
X.shape

# %%
Y.shape

# %%
X *= 200

# %%
os.makedirs(os.path.expanduser("~/project/51_1_hattie_data/"), exist_ok=True)

# %%
# # create real data

# %%
pd.DataFrame(X.numpy()).to_csv(f"~/project/51_1_hattie_data/X_train_0.csv", index=False, header=False)
# Split Y_train and K_test into individual parts
for j in range(len(Y[0])):
    # print(j)
    Y_train_j=Y[:, j]
    pd.DataFrame(Y_train_j.numpy()).to_csv(f"~/project/51_1_hattie_data/Y_train_0_{j}.csv", index=False, header=False)

