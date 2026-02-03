# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/16_atresia_markers.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # This notes aims to identify the genes that are biomarkers for atresia.  Hopefully, they include ```Ghr```, ```Cfh```, and ```Pik3r1```.

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
file_path = os.path.expanduser('~/project/mouse_ovary_slide_seq_young_estrus.h5ad')
adata = sc.read_h5ad(file_path)

# %%
adata.obs.columns.tolist()

# %%
adata.obs['new_annot_foll']

# %%
adata.obs['new_annot_foll'].unique().tolist()

# %%
adata.obs['new_annot_foll'][adata.obs['new_annot_foll'] == "GC - atretic"]

# %%
# ## In the following, we look for the highly expressed genes in the atretic cells, and perform DE analysis to identify the markers for atresia.

# %%
file_path = os.path.expanduser('~/project/mouse_ovary_slide_seq_young_estrus.h5ad')
adata = sc.read_h5ad(file_path)

# %%
# Calculate total expression for each cell in adata.raw
total_expression = adata.raw.X.sum(axis=1).A1  # Use .A1 to flatten the sparse matrix result

print(total_expression[:20])

# %%
adata.raw.X.shape

# %%
adata.X.shape

# %%
pass

# %%
adata.raw.X[0:10,0:10].todense()

# %%
adata.X[0:10, 0:10].todense()

# %%
# Calculate total expression for each cell in adata
total_expression = adata.X.sum(axis=1).A1  # Use .A1 to flatten the sparse matrix result

# View the updated adata.obs
print(total_expression[:20])

# Add the total expression as a new column in adata.obs
adata.obs['total_expression'] = total_expression
print(adata.obs[['total_expression']].head(20))

# %%
pass

# %%
# ## Certainly, ```adata``` looks different from ```adata.raw```, I am confused on what kind of preprocessing step was used, but perhaps not the usual normalisation or log1p, since even the relative rankings of total cell expressions are different.  What happened during preprocessing??

# %%
# Step 1: Data Preview
# ----------------------------------------
print("data preview:")
print(adata.X[:5, :5])  # Preview a small slice of the data matrix

# Step 2: Subset AnnData to Atretic Cells
# ----------------------------------------
# Extract only the cells labeled as "GC - atretic" from `adata.obs['new_annot_foll']`
atretic_cells = adata[adata.obs['new_annot_foll'] == "GC - atretic"]

# Step 3: Compute Mean Gene Expression in Atretic Cells
# -----------------------------------------------------
# Calculate the mean expression for each gene across these cells
mean_expression = atretic_cells.X.mean(axis=0).A1  # Use `.A1` to handle sparse matrix
gene_names = adata.var_names  # Retrieve gene names

# Create a DataFrame with mean expression for easier handling
mean_expression_df = pd.DataFrame({
    'gene': gene_names,
    'mean_expression': mean_expression
}).sort_values(by='mean_expression', ascending=False)

# Save or display the top N highly expressed genes
print("Top 10 highly expressed genes in atretic cells:")
print(mean_expression_df.head(10))  # Top 10 genes

# Step 4: Differential Gene Expression Analysis
# ---------------------------------------------
# Perform differential expression analysis to identify markers
sc.tl.rank_genes_groups(adata, groupby='new_annot_foll', groups=['GC - atretic'], reference='rest')

# Plot the top marker genes
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)  # Top 20 marker genes

# Retrieve differential expression results for "GC - atretic"
de_results = pd.DataFrame({
    'gene': adata.uns['rank_genes_groups']['names']['GC - atretic'],
    'logfoldchange': adata.uns['rank_genes_groups']['logfoldchanges']['GC - atretic'],
    'pval': adata.uns['rank_genes_groups']['pvals']['GC - atretic'],
    'pval_adj': adata.uns['rank_genes_groups']['pvals_adj']['GC - atretic']  # Adjusted p-values
})

# Save or display the top biomarkers
print("Top 10 differentially expressed genes in GC - atretic cells:")
print(de_results.head(10))  # Top 10 markers

# Step 5: Visualization of Top Biomarkers
# ---------------------------------------
# Select the top 5 biomarkers for detailed visualization
top_genes = de_results.head(5)['gene'].tolist()

# Plot violin plots for the top genes
sc.pl.violin(adata, top_genes, groupby='new_annot_foll', stripplot=False, jitter=False)

# Plot dot plot for the top genes
sc.pl.dotplot(adata, top_genes, groupby='new_annot_foll')

# Step 6: Save Results for Further Analysis
# -----------------------------------------
# Save the results to CSV for further exploration or reporting
mean_expression_df.to_csv("mean_expression_atretic_cells.csv", index=False)
de_results.to_csv("differential_expression_atretic_cells.csv", index=False)

print("Analysis complete. Results saved as CSV files.")

# %%
# ## ```Cfh``` shows up once as 9th place in the first list, but does not show up afterwards.  Strangely, ```Ghr``` and ```Pik3r1``` are nowhere to be found!

# %%
# ## Maybe we need to instead do the analysis on ```adata.raw```.  Let's to see if different results show up.

# %%
import scanpy as sc
import pandas as pd

# Step 2: Subset AnnData to Atretic Cells (Using adata.raw)
# ---------------------------------------------------------
# Extract only the cells labeled as "GC - atretic" from `adata.obs['new_annot_foll']`
atretic_cells = adata[adata.obs['new_annot_foll'] == "GC - atretic"].raw

# Step 3: Compute Mean Gene Expression in Atretic Cells
# -----------------------------------------------------
# Calculate the mean expression for each gene across these cells
mean_expression = atretic_cells.X.mean(axis=0).A1  # Use `.A1` for sparse matrix handling
gene_names = adata.raw.var_names  # Retrieve gene names from adata.raw

# Create a DataFrame with mean expression for easier handling
mean_expression_df = pd.DataFrame({
    'gene': gene_names,
    'mean_expression': mean_expression
}).sort_values(by='mean_expression', ascending=False)

# Save or display the top N highly expressed genes
print("Top 10 highly expressed genes in atretic cells:")
print(mean_expression_df.head(10))  # Top 10 genes

# Step 4: Differential Gene Expression Analysis (Using adata.raw)
# ---------------------------------------------------------------
# Perform differential expression analysis to identify markers
# For this, we still use the main adata object but rely on raw values
sc.tl.rank_genes_groups(adata, groupby='new_annot_foll', groups=['GC - atretic'], reference='rest', use_raw=True)

# Plot the top marker genes
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)  # Top 20 marker genes

# Retrieve differential expression results for "GC - atretic"
de_results = pd.DataFrame({
    'gene': adata.uns['rank_genes_groups']['names']['GC - atretic'],
    'logfoldchange': adata.uns['rank_genes_groups']['logfoldchanges']['GC - atretic'],
    'pval': adata.uns['rank_genes_groups']['pvals']['GC - atretic'],
    'pval_adj': adata.uns['rank_genes_groups']['pvals_adj']['GC - atretic']  # Adjusted p-values
})

# Save or display the top biomarkers
print("Top 10 differentially expressed genes in GC - atretic cells:")
print(de_results.head(10))  # Top 10 markers

# Step 5: Visualization of Top Biomarkers
# ---------------------------------------
# Select the top 5 biomarkers for detailed visualization
top_genes = de_results.head(5)['gene'].tolist()

# Plot violin plots for the top genes
sc.pl.violin(adata, top_genes, groupby='new_annot_foll', stripplot=False, jitter=False, use_raw=True)

# Plot dot plot for the top genes
sc.pl.dotplot(adata, top_genes, groupby='new_annot_foll', use_raw=True)

# Step 6: Save Results for Further Analysis
# -----------------------------------------
# Save the results to CSV for further exploration or reporting
mean_expression_df.to_csv("mean_expression_atretic_cells_raw.csv", index=False)
de_results.to_csv("differential_expression_atretic_cells_raw.csv", index=False)

print("Analysis complete. Results saved as CSV files.")

# %%
# # Unfortunately, the genes ```Ghr``` and ```Pik3r1``` are once again nowhere to be found!

