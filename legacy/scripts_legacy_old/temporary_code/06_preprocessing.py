# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/06_preprocessing.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Figure out the right way to preprocess

# %%
import numpy as np

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
# def load_data(gene_list, head=0, puck_list = "all"):
#     # Load the H5AD file
#     file_path = os.path.expanduser('~/project/mouse_ovary_slide_seq_young_estrus.h5ad')
#     adata = sc.read_h5ad(file_path)
#     if puck_list != "all":
#         adata = adata[adata.obs['puck'].isin(puck_list)]
        
#     # Extract spatial coordinates
#     coordinates = pd.DataFrame(adata.obsm["spatial"], columns=['x', 'y'], index=adata.obs_names)
#     # Extract gene expression levels
#     gene_data = [isolate_gene_values(adata, gene) for gene in gene_list]
#     # Combine to create df
#     df = pd.concat([coordinates] + gene_data, axis=1)

#     # Shuffle and sample the DataFrame
#     df = df.sample(frac=1)
#     if head!=0:
#         df = df.head(head)
    
#     # Preprocessing Step 1: Identify columns that sum to 0; drop those columns from df and from gene_list; divide the rest by their sums
#     column_sums = df[gene_list].sum()
#     zero_sum_columns = column_sums[column_sums == 0].index
#     if len(zero_sum_columns) > 0:
#         print(f"{list(zero_sum_columns)} dropped from gene_list due to not being observed.")
#         column_sums = column_sums.drop(zero_sum_columns)
#         df = df.drop(columns=zero_sum_columns)
#         gene_list = [col for col in gene_list if col not in zero_sum_columns]
#     # Divide each column by its corresponding sum
#     df[gene_list] = df[gene_list].div(column_sums, axis=1)

#     # Preprocessing Step 2: Divide 'x' and 'y' by 5000; log(1+x) the genes
#     df['x'] = df['x'] / 5000
#     df['y'] = df['y'] / 5000
#     for gene in gene_list:
#         df[gene] = np.log(df[gene]+1)

#     # Preprocessing Step 3: Make all standard deviations equal to 100
#     for gene in gene_list:
#         std = df[gene].std()
#         if std == 0:
#             print(f"error: {gene} has zero standard deviation")
#         else:
#             df[gene] = df[gene] / std * 10

#     # Output Objects
#     # Convert 'x' and 'y' columns into a tensor for spatial coordinates, X
#     X = torch.tensor(df[['x', 'y']].values, dtype=torch.float64)
#     # Convert gene columns into a tensor for gene expression values, Y
#     Y = torch.tensor(df[gene_list].values, dtype=torch.float64)
#     return adata, X, Y, df, gene_list

# %%
# # Global parameters:
number_of_cycles = 1 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 1
dims = 2  # 2D spatial
head = 0 # how many locations to consider in the real data set.

# %%
gene_list = ["Inha", "Esr1"]
gene_list = ["Inha"]
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
puck_list = ['Puck_230517_39'] # the largest puck
adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
df

# %%
# Plot scatter plots for each gene in gene_list
for gene in gene_list:
    plt.figure(figsize=(6, 4))
    plt.hist(df[gene])
    plt.title(f"histogram of {gene}")
    plt.show()

# %%
# Plot scatter plots for each gene in gene_list
for gene in gene_list:
    plt.figure(figsize=(6, 4))
    plt.scatter(df['x'], df['y'], c=df[gene], cmap='viridis', s=.05)
    plt.colorbar(label=f'{gene} Expression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Spatial Distribution of {gene}')
    plt.show()

