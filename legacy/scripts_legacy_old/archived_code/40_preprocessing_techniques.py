# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/archived_code/40_preprocessing_techniques.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Compare different preprocessing strategies

# %%
%run -i preambles
%run -i helper_functions
%run -i fitting_functions
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

gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr"]

puck_list = ['Puck_230517_39'] # the largest puck
adata, X,Y,df=load_data(gene_list=gene_list, head=head, puck_list = puck_list)
df

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Log transformation
for gene in gene_list:
    df[gene + '_log'] = np.log(df[gene] + 1)

# Plot before and after log transformation for all genes
fig, axes = plt.subplots(nrows=len(gene_list), ncols=2, figsize=(12, len(gene_list) * 4))

for i, gene in enumerate(gene_list):
    axes[i, 0].hist(df[gene], bins=30, alpha=0.7)
    axes[i, 0].set_title(f"Histogram of {gene} (Original)")
    
    axes[i, 1].hist(df[gene + '_log'], bins=30, alpha=0.7)
    axes[i, 1].set_title(f"Histogram of {gene} (Log Transformed)")

plt.tight_layout()
plt.show()

# Perform Shapiro-Wilk test for normality
for gene in gene_list:
    stat, p = stats.shapiro(df[gene + '_log'])
    print(f'{gene}_log: Shapiro-Wilk Test: Stat={stat}, p-value={p}')

# %%
# Z-score normalization
for gene in gene_list:
    df[gene + '_zscore'] = (df[gene] - df[gene].mean()) / df[gene].std()

# Plot before and after z-score normalization for all genes
fig, axes = plt.subplots(nrows=len(gene_list), ncols=2, figsize=(12, len(gene_list) * 4))

for i, gene in enumerate(gene_list):
    axes[i, 0].hist(df[gene], bins=30, alpha=0.7)
    axes[i, 0].set_title(f"Histogram of {gene} (Original)")
    
    axes[i, 1].hist(df[gene + '_zscore'], bins=30, alpha=0.7)
    axes[i, 1].set_title(f"Histogram of {gene} (Z-Score Normalized)")

plt.tight_layout()
plt.show()

# Perform Shapiro-Wilk test for normality
for gene in gene_list:
    stat, p = stats.shapiro(df[gene + '_zscore'])
    print(f'{gene}_zscore: Shapiro-Wilk Test: Stat={stat}, p-value={p}')

# %%
def quantile_normalize(df_input):
    sorted_df = np.sort(df_input.values, axis=0)
    rank_mean = sorted_df.mean(axis=1)
    df_output = df_input.rank(method="min").stack().astype(int).map(dict(zip(range(1, len(rank_mean)+1), rank_mean))).unstack()
    return df_output

# Apply quantile normalization
df_quantile_normalized = quantile_normalize(df[gene_list])

# Plot before and after quantile normalization for all genes
fig, axes = plt.subplots(nrows=len(gene_list), ncols=2, figsize=(12, len(gene_list) * 4))

for i, gene in enumerate(gene_list):
    axes[i, 0].hist(df[gene], bins=30, alpha=0.7)
    axes[i, 0].set_title(f"Histogram of {gene} (Original)")
    
    axes[i, 1].hist(df_quantile_normalized[gene], bins=30, alpha=0.7)
    axes[i, 1].set_title(f"Histogram of {gene} (Quantile Normalized)")

plt.tight_layout()
plt.show()

# Perform Shapiro-Wilk test for normality
for gene in gene_list:
    stat, p = stats.shapiro(df_quantile_normalized[gene])
    print(f'{gene}_quantile_normalized: Shapiro-Wilk Test: Stat={stat}, p-value={p}')

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax_scaled = df.copy()

# Apply min-max scaling
df_minmax_scaled[gene_list] = scaler.fit_transform(df[gene_list])

# Plot before and after min-max scaling for all genes
fig, axes = plt.subplots(nrows=len(gene_list), ncols=2, figsize=(12, len(gene_list) * 4))

for i, gene in enumerate(gene_list):
    axes[i, 0].hist(df[gene], bins=30, alpha=0.7)
    axes[i, 0].set_title(f"Histogram of {gene} (Original)")
    
    axes[i, 1].hist(df_minmax_scaled[gene], bins=30, alpha=0.7)
    axes[i, 1].set_title(f"Histogram of {gene} (Min-Max Scaled)")

plt.tight_layout()
plt.show()

# Perform Shapiro-Wilk test for normality
for gene in gene_list:
    stat, p = stats.shapiro(df_minmax_scaled[gene])
    print(f'{gene}_minmax_scaled: Shapiro-Wilk Test: Stat={stat}, p-value={p}')

# %%
# Apply square root transformation as a proxy for VST
for gene in gene_list:
    df[gene + '_sqrt'] = np.sqrt(df[gene])

# Plot before and after square root transformation for all genes
fig, axes = plt.subplots(nrows=len(gene_list), ncols=2, figsize=(12, len(gene_list) * 4))

for i, gene in enumerate(gene_list):
    axes[i, 0].hist(df[gene], bins=30, alpha=0.7)
    axes[i, 0].set_title(f"Histogram of {gene} (Original)")
    
    axes[i, 1].hist(df[gene + '_sqrt'], bins=30, alpha=0.7)
    axes[i, 1].set_title(f"Histogram of {gene} (VST Approximation)")

plt.tight_layout()
plt.show()

# Perform Shapiro-Wilk test for normality
for gene in gene_list:
    stat, p = stats.shapiro(df[gene + '_sqrt'])
    print(f'{gene}_sqrt: Shapiro-Wilk Test: Stat={stat}, p-value={p}')

# %%
pass

# %%
pass

# %%
pass

# %%
pass

