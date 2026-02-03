# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/08_quantisation_of_dist.ipynb on 2026-02-03 10:30:29
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
head= 0
puck_list = ['Puck_230517_39'] # the largest puck
# gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", 
    # "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr", 
    # "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
gene_list = ["Inha"]
adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
# ## Summary Statistics about torch.cdist(X,X)

# %%
X_dist = torch.cdist(X,X)
# Step 1: Create a boolean mask that identifies the off-diagonal elements
off_diagonal_mask = ~torch.eye(X_dist.size(0), dtype=bool)

# Step 2: Use the mask to extract only the off-diagonal elements
off_diagonal_elements = X_dist[off_diagonal_mask]

# Step 3: Flatten the off-diagonal elements
off_diagonal = off_diagonal_elements.flatten()

flat_matrix = torch.flatten(off_diagonal)
df = pd.DataFrame(flat_matrix)
df.describe()

# %%
df.max()/df.min()

# %%
# This shows that the furtherest distnace between two points and the closest distance two points are only 365 times apart from each other

# %%
# ## Let us do 800 locations, so that histogram plotting is possible

# %%
head = 800 # how many locations to consider in the real data set.
adata, X,Y,df, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)
df

# %%
X_dist = torch.cdist(X,X)
# Step 1: Create a boolean mask that identifies the off-diagonal elements
off_diagonal_mask = ~torch.eye(X_dist.size(0), dtype=bool)

# Step 2: Use the mask to extract only the off-diagonal elements
off_diagonal_elements = X_dist[off_diagonal_mask]

# Step 3: Flatten the off-diagonal elements
off_diagonal = off_diagonal_elements.flatten()

flat_matrix = torch.flatten(off_diagonal)
len(flat_matrix)

# %%
# Plot the histogram directly using the PyTorch tensor
plt.hist(flat_matrix, bins=100)
plt.title('Histogram of Flattened Matrix')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# %%
plt.scatter(X[:,0], X[:,1], s=8)

# %%
# ## An approximate method for computing K

# %%
alpha_i = torch.tensor(0.01, dtype=torch.float64, device=device).requires_grad_(True)
nu_i = torch.tensor(1.0, dtype=torch.float64, device=device).requires_grad_(True)
sigma_i = torch.tensor(1.0, dtype=torch.float64, device=device).requires_grad_(True)
K_approx =approx_matern_kernel_marginal(X, nu_i, alpha_i, sigma_i, epsilon=1e-9)

# %%
K = matern_kernel(torch.cdist(X, X), nu_i, alpha_i, sigma_i)

# %%
torch.norm(K, p='fro')**2

# %%
torch.norm(K_approx, p='fro')**2

# %%
torch.norm(K_approx - K, p='fro')**2

# %%
(torch.norm(K_approx - K, p='fro') / torch.norm(K, p='fro'))**2

# %%
# ## Do the same for the cross fitting function

# %%
# Define test parameters
p = 2  # Number of parameters (for alpha, nu, sigma matrices)
n_locations = 3  # Number of locations
dimensions = 2  # Dimensionality of the locations

# Randomly generate test inputs
X = torch.randn(n_locations, dimensions)
alpha_matrix = torch.rand(p, p)
nu_matrix = torch.rand(p, p)
sigma_matrix = torch.rand(p, p)

# Exact computation using compute_matern_covariance
exact_covariance = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)

# Approximate computation using approx_matern_kernel_cross
approx_covariance = approx_matern_kernel_cross(nu_matrix, alpha_matrix, sigma_matrix, X)

# Compare the Frobenius norm of the difference between the exact and approximate covariance matrices
frobenius_norm_difference = (torch.norm(exact_covariance - approx_covariance, p='fro')**2) / (torch.norm(exact_covariance, p='fro')**2)

print(f"noramlised Frobenius norm of the difference: {frobenius_norm_difference.item()}")

# Set a tolerance level to check if the approximation is sufficiently close
tolerance = 1e-2
assert frobenius_norm_difference < tolerance, "The approximation is not close enough to the exact solution"
print("Test passed: The approximate covariance matrix is sufficiently close to the exact one.")

# %%
torch.norm(exact_covariance - approx_covariance, p='fro')**2

# %%
torch.norm(exact_covariance, p='fro')**2

# %%
torch.norm(approx_covariance, p='fro')**2

# %%
exact_covariance

# %%
approx_covariance

# %%
# ## Compute the pairwise distance histogram for a regular 15 by 15 or 30 by 30 grid.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def create_grid_and_plot_histogram(k):
    # Create a regular k by k grid of points
    x = np.linspace(0, k, k)
    y = np.linspace(0, k, k)
    
    # Create a meshgrid of x and y coordinates
    xx, yy = np.meshgrid(x, y)
    
    # Flatten the grid to get a list of 2D points
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Compute all pairwise distances using Euclidean distance
    distances = pdist(points, metric='euclidean')
    
    # Plot the histogram of the pairwise distances
    plt.hist(distances, bins=20, edgecolor='black')
    plt.title(f'Histogram of Pairwise Distances for k = {k}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()

# Plot for k = 15
create_grid_and_plot_histogram(15)

# Plot for k = 30
create_grid_and_plot_histogram(30)

# %%
pass

# %%
pass

