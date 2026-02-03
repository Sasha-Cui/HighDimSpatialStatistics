# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/archived_code/31_ConvertToRData.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # I convert Hattie Chung's data set into an .RData file.

# %%
from preamble import *

# %%
# # Global parameters:
number_of_cycles = 500 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
locations_per_group = 200 # how many locations to observe per group
number_of_locations = number_of_groups * locations_per_group # total locations
number_of_simulations = 50 # for synthetic data, how many optimisation to average over
steps_per_batch = 5
dims = 2  # 2D spatial
p =  3 # how many features 

# %%
def isolate_gene_values (adata, gene_name):
    gene_values = pd.DataFrame(adata[:, gene_name].X.toarray(), columns=[gene_name], index=adata.obs_names)
    return gene_values

head=1500
adata = sc.read_h5ad('ovary_Puck_230517_39.h5ad')
coordinates = pd.DataFrame(adata.obsm["spatial"], columns=['x', 'y'], index=adata.obs_names)
df = pd.concat([
    coordinates,
    isolate_gene_values(adata, "Serpine2"),
    isolate_gene_values(adata, "Tagln"),
    isolate_gene_values(adata, "Acta2"),
    isolate_gene_values(adata, "Mgp"),
    isolate_gene_values(adata, "S100a6"),
    isolate_gene_values(adata, "Col1a2"),
    isolate_gene_values(adata, "Nr5a2"),
    isolate_gene_values(adata, "Inhba"),
    isolate_gene_values(adata, "Tpm2"),
    isolate_gene_values(adata, "Tdrd5")
], axis=1)
df = df.sample(frac=1)
df = df.head(head)
df['x'] = df['x'] / df['x'].median()
df['y'] = df['y'] / df['y'].median()

# Normalize all gene columns by dividing by 1000
genes_to_include = ["Serpine2", "Tagln", "Acta2", "Mgp", "S100a6", "Col1a2", "Nr5a2", "Inhba", "Tpm2", "Tdrd5"]
for gene in genes_to_include:
    df[gene] = df[gene] / 1000
df = df.reset_index(drop=True)
# Create a dictionary mapping gene names to numbers
gene_mapping = {
    "Serpine2": 1,
    "Tagln": 2,
    "Acta2": 3,
    "Mgp": 4,
    "S100a6": 5,
    "Col1a2": 6,
    "Nr5a2": 7,
    "Inhba": 8,
    "Tpm2": 9,
    "Tdrd5": 10
}
df

# %%
df = pd.melt(df, id_vars=['x', 'y'], var_name='Gene', value_name='Expression')
df = df[['Expression', 'x','y','Gene']]
df

# %%
# Replace the gene names with numbers
df['Gene'] = df['Gene'].map(gene_mapping)
df

# %%
import subprocess
# Assume df is your pandas DataFrame
df.to_csv('temp_data.csv', index=False)

# R script to convert CSV to RData
r_script = """
data <- read.csv('temp_data.csv')
save(data, file='data_matrix.RData')
"""

# Run the R script using subprocess
subprocess.run(['Rscript', '-e', r_script])

# %%
pass

