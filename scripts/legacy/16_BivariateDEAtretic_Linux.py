# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/16_BivariateDEAtretic_Linux.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# **Atresia** refers to the process by which ovarian follicles degenerate and are reabsorbed before reaching maturity. This is a natural part of the ovarian cycle in females and affects the majority of follicles within the ovaries.
#
# ### Detailed Explanation:
#
# 1. **Ovarian Follicles**:
#    - **Definition**: Ovarian follicles are small fluid-filled sacs within the ovaries that contain immature eggs (oocytes). Each follicle nurtures an oocyte and supports its development.
#    - **Follicular Development**: Follicles go through several stages of development: primordial, primary, secondary, and tertiary (or antral). Only a small number of follicles reach the later stages of development.
#
# 2. **Atresia**:
#    - **Definition**: Atresia is the process of degeneration and reabsorption of ovarian follicles that do not complete their development. This process can occur at any stage of follicular development.
#    - **Mechanism**: Atresia involves apoptosis (programmed cell death) of the follicular cells and the oocyte. The breakdown products are then phagocytosed by surrounding cells and reabsorbed into the ovarian tissue.
#
# 3. **Significance of Atresia**:
#    - **Follicular Reserve**: Females are born with a finite number of primordial follicles, which constitute the ovarian reserve. Throughout a woman’s reproductive life, many of these follicles undergo atresia, with only a fraction reaching ovulation.
#    - **Regulation**: Atresia is regulated by hormonal signals and intra-ovarian factors. Key hormones involved include follicle-stimulating hormone (FSH) and luteinizing hormone (LH). Lower levels of FSH, for instance, can lead to increased atresia of developing follicles.
#    - **Selection**: The process of atresia allows for the selection of the healthiest follicles for ovulation. Typically, only one follicle (in humans) becomes the dominant follicle that is capable of ovulating, while the others undergo atresia.
#
# 4. **Clinical Relevance**:
#    - **Fertility**: The rate of follicular atresia has implications for fertility. A higher rate of atresia can reduce the number of available eggs for ovulation, potentially impacting fertility.
#    - **Aging**: As women age, the number of follicles that can mature and ovulate decreases, partly due to the cumulative effect of atresia. This contributes to the decline in fertility with age.
#    - **Polycystic Ovary Syndrome (PCOS)**: In conditions like PCOS, atresia may be altered, leading to the presence of multiple immature follicles and hormonal imbalances that affect ovulation and fertility.
#
# ### Summary:
#
# **Atresia** in the context of ovarian cells refers to the natural process of follicular degeneration and reabsorption, which prevents many ovarian follicles from reaching maturity and ovulation. This process is a key component of ovarian physiology, regulating the selection of follicles for ovulation and maintaining the ovarian reserve. Understanding atresia is important in reproductive biology and has significant implications for fertility and ovarian health.

# %%
from datetime import datetime
str(datetime.now())

# %%
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd

# %%
# Load the H5AD file
adata_original = sc.read_h5ad('ovary_Puck_230517_39.h5ad')

# %%
adata = adata_original

# %%
# # Basic exploratory analysis of the data

# %%
# Check the content of 'spatial' to ensure it exists and is in the correct format
if 'spatial' in adata.obsm.keys():
    print("Spatial coordinates found in adata.obsm['spatial']:")
    print(adata.obsm['spatial'])
else:
    print("No spatial coordinates found in adata.obsm['spatial']")

# Provide spot_size directly if .uns['spatial'][library_id] does not exist
sc.pl.spatial(
    adata,
    color='cell_type_major',  # You can specify a column in adata.obs to color by if desired
    basis='spatial',  # Specify the key in obsm to use for spatial coordinates
    spot_size=50,  # Adjust this value as necessary for your data
    show=True,
)

# %%
column = "segment_label_atresia"
values = adata.obs[column]
plt.figure(figsize=(10, 6))
values.value_counts().plot(kind='bar', edgecolor='black')
plt.title(f'Histogram of {column} Values')
plt.xlabel(column)
plt.ylabel('Frequency')
plt.show()

# %%
# An important notice here is to not use the raw data set.  Otherwise, you would have been looking at genes that have been excluded by earlier pre-processing. 

# %%
# # Differential Expression Analysis

# %%
# Perform DEG analysis
sc.tl.rank_genes_groups(adata, 'segment_label_atresia', groups=['partially atretic', 'other'], reference='non-atretic', method='t-test',  use_raw=False)

# Extract and display the results
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
degs = pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
     for group in groups for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj']}
)

print(degs.head())

# Visualize the top differentially expressed genes
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# %%
# Perform DEG analysis
sc.tl.rank_genes_groups(adata, 'segment_label_atresia', groups=['non-atretic', 'other'], reference='partially atretic', method='t-test',use_raw=False)

# Extract and display the results
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
degs = pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
     for group in groups for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj']}
)

print(degs.head())

# Visualize the top differentially expressed genes
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# %%
# Perform DEG analysis
sc.tl.rank_genes_groups(adata, 'segment_label_atresia', groups=['non-atretic', 'partially atretic'], reference='other', method='t-test', use_raw=False)

# Extract and display the results
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
degs = pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
     for group in groups for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj']}
)

print(degs.head())

# Visualize the top differentially expressed genes
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# %%
# # Isolate Gene Expression Values and Writing into a data frame

# %%
def isolate_gene_values (gene_name):
    gene_values = pd.DataFrame(adata[:, gene_name].X.toarray(), columns = [gene_name], index = adata.obs_names)
    return gene_values

# %%
coordinates = pd.DataFrame(adata.obsm["spatial"], columns = ['x','y'], index = adata.obs_names)

# %%
df = pd.concat([coordinates, isolate_gene_values("Serpine2"),isolate_gene_values("Tagln")], axis =1)

# %%
df=df.sample(frac=1)
df=df.head(8000)

# %%
print(len(isolate_gene_values("Tagln")["Tagln"].unique()))
print(len(isolate_gene_values("Serpine2")["Serpine2"].unique()))

# %%
# SerpinE2 is required in the extracellular milieu of breast tumors where it acts in multiple ways to regulate tumor matrix deposition, thereby controlling tumor cell dissemination.
#
# The Tagln gene encodes a shape change and transformation sensitive actin-binding protein which belongs to the calponin family. It is ubiquitously expressed in vascular and visceral smooth muscle, and is an early marker of smooth muscle differentiation. The encoded protein is thought to be involved in calcium-independent smooth muscle contraction. It acts as a tumor suppressor, and the loss of its expression is an early event in cell transformation and the development of some tumors, coinciding with cellular plasticity. 

# %%
# # From Python DataFrame to R List Vector

# %%
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, ListVector
from rpy2.robjects import pandas2ri
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects import r
# Import necessary R packages
geoR = importr('geoR')

pandas2ri.activate()
robjects.numpy2ri.activate()

# %%
# Converting from pandas DataFrame to ListVector
def df_to_list_vector (df, gene_name):
    
    # Extract the coordinates as a numpy array
    coords = df[['x', 'y']].to_numpy()
    # Turn it into a float array
    r_coords = numpy2rpy(coords)
    
    # Extract the gene expression values as a float vector
    r_gene_expression = FloatVector(df[gene_name])
    
    # Create the R list vector
    r_list = ListVector({
        'coords': r_coords,
        'data': r_gene_expression,
        'gene name': gene_name
        
    })
    
    # Print the R list vector
    return(r_list)

s200 = df_to_list_vector(df,gene_name = "Tagln")
print(s200)

# %%
# Assign s200 to the R environment
robjects.globalenv['s200'] = s200

# %%
%%time 
r_code = """
ml <- likfit(s200, cov.model = 'matern', kappa = 1.5, ini = c(0.9, 0.2),nug = 0.2)
ml
"""
result = r(r_code)
print(result)

# %%
%%time
r_code = """
ml <- likfit(s200, cov.model = 'exponential', kappa = 1.5, ini = c(0.9, 0.2), nug = 0.2)
ml
"""
result = r(r_code)
print(result)

# %%
s200 = df_to_list_vector(df,gene_name = "Serpine2")

# %%
print(s200)

# %%
# Assign s200 to the R environment
robjects.globalenv['s200'] = s200

# %%
%%time 
r_code = """
ml <- likfit(s200, cov.model = 'matern', kappa = 1.5, ini = c(0.9, 0.2),nug = 0.2)
ml
"""
result = r(r_code)
print(result)

# %%
%%time
r_code = """
ml <- likfit(s200, cov.model = 'exponential', kappa = 1.5, ini = c(0.9, 0.2), nug = 0.2)
ml
"""
result = r(r_code)
print(result)

# %%
pass

# %%
# # From R ListVector to Python DataFrame

# %%
# Converting from ListVector to pandas DataFrame
# Example data: using 's100' dataset from geoR package
robjects.r('data(s100)')
s100=robjects.r('s100')

df1 = pd.concat([pd.DataFrame(list(s100[0][:,0]), columns = ['x']), pd.DataFrame(list(s100[0][:,1]), columns = ['y']), pd.DataFrame(list(s100[1]), columns = ['z'])], axis=1)
df1

# %%
robjects.r['pi'][0]

# %%
from datetime import datetime
str(datetime.now())

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

# %%
pass

