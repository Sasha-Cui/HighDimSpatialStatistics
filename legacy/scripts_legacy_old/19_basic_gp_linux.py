# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/19_basic_gp_linux.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
import GPy
from datetime import datetime
str(datetime.now())

# %%
# Load the H5AD file
adata_original = sc.read_h5ad('ovary_Puck_230517_39.h5ad')

adata = adata_original.copy()

# Isolate the spatial coordinates
coordinates = pd.DataFrame(adata.obsm["spatial"], columns = ['x','y'], index = adata.obs_names)

# Isolate the most explainatory genes
def isolate_gene_values (gene_name):
    gene_values = pd.DataFrame(adata[:, gene_name].X.toarray(), columns = [gene_name], index = adata.obs_names)
    return gene_values
var1_name = "Serpine2"
var2_name = "Tagln"

# Create the df with coordinates and biomarkers of interest
df = pd.concat([coordinates, isolate_gene_values(var1_name),isolate_gene_values(var2_name)], axis =1)


# df.replace(0, pd.NA, inplace=True)
df = df.dropna()
df = df.sample(frac=1)

df = df.head(6000)

df['x'] = df['x']/df['x'].median()
df['y'] = df['y']/df['y'].median()
# df[var1_name] = df[var1_name]/df[var1_name].median()
# df[var2_name] = df[var2_name]/df[var2_name].median()

# %%
df

# %%
# As of Mon 12th of Oct running on devel branch of GPy 0.8.8

# %%
# plotting instructions: it is better to not use the following line
# GPy.plotting.change_plotting_library('plotly')

# only write m.plot() for plots.  That is enough.

# %%
# # Gaussian process regression tutorial
#
# ### Nicolas Durrande 2013
# #### with edits by James Hensman and Neil D. Lawrence
#
# We will see in this tutorial the basics for building a 1 dimensional and a 2 dimensional Gaussian process regression model, also known as a kriging model.
#
# We first import the libraries we will need:

# %%
# ## 2-dimensional example
#
# Here is a 2 dimensional example:

# %%
%%time 
# Extract coordinates and variables
X = df[['x', 'y']].values
Y = df[[var1_name, var2_name]].values

# define kernel
ker = GPy.kern.Matern52(input_dim=2,ARD=True) + GPy.kern.White(2)

# create simple GP model
m = GPy.models.GPRegression(X,Y,ker)

# optimize and plot
m.optimize(messages=True)

# %%
m.param_array

# %%
# fig = m.plot()
# m.plot()
m.optimize_restarts(num_restarts = 10)

# %%
# # 1: Saving a model:
# np.save('model_save.npy', m.param_array)

# # 2: loading a model
# np.load('model_save.npy')
m.param_array

# %%
# The flag `ARD=True` in the definition of the `Matern` kernel specifies that we want one lengthscale parameter per dimension (ie the GP is not isotropic). Note that for 2-d plotting, only the mean is shown. 

# %%
X_new = np.random.uniform(-2., 2., (500, 2))
Y_pred, Y_var = m.predict(X_new)

# %%
Y_pred[:,1]

# %%
# Setup plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot for the first variable
sc1 = axs[0].scatter(X_new[:, 0], X_new[:, 1], c=Y_pred[:, 0], cmap='viridis', edgecolor='k', s=50)
fig.colorbar(sc1, ax=axs[0], label='Value of Variable 1')
axs[0].set_title('Variable 1 Predictions')
axs[0].set_xlabel('X_new[:, 0]')
axs[0].set_ylabel('X_new[:, 1]')

# Plot for the second variable
sc2 = axs[1].scatter(X_new[:, 0], X_new[:, 1], c=Y_pred[:, 1], cmap='viridis', edgecolor='k', s=50)
fig.colorbar(sc2, ax=axs[1], label='Value of Variable 2')
axs[1].set_title('Variable 2 Predictions')
axs[1].set_xlabel('X_new[:, 0]')
axs[1].set_ylabel('X_new[:, 1]')

# Show plot
plt.tight_layout()
plt.show()

# %%
# Setup plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot for the first variable
sc1 = axs[0].scatter(df['x'], df['y'], c=df[var1_name], cmap='viridis', edgecolor='k', s=5)
fig.colorbar(sc1, ax=axs[0], label='Value of Variable 1')
axs[0].set_title('Variable 1 Data')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# Plot for the second variable
sc1 = axs[0].scatter(df['x'], df['y'], c=df[var2_name], cmap='viridis', edgecolor='k', s=500)
fig.colorbar(sc2, ax=axs[1], label='Value of Variable 2')
axs[1].set_title('Variable 2 Data')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

# Show plot
plt.tight_layout()
plt.show()

# %%
df[var1_name].hist(bins=20)

# %%
df[var2_name].hist(bins=20)

# %%
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

