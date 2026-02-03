##helper_functions 
# 25 Oct 2024: made sure everything is in the order of alpha, nu, sigma.
def isolate_gene_values(adata, gene_name):
    # Extract gene expression from adata.raw and convert to torch tensor
    gene_data = adata.raw[:, gene_name].X.toarray()  # Assuming .X is in sparse format
    return torch.tensor(gene_data, dtype=torch.float64, device=device)

# Load and preprocess data directly on GPU
def load_data(gene_list, head=0, puck_list="all"):
    # Load the H5AD file
    file_path = os.path.expanduser('~/project/mouse_ovary_slide_seq_young_estrus.h5ad')
    adata = sc.read_h5ad(file_path)
    
    if puck_list != "all":
        adata = adata[adata.obs['puck'].isin(puck_list)]
    
    # Extract spatial coordinates and move to GPU
    coordinates = torch.tensor(adata.obsm["spatial"], dtype=torch.float64, device=device)

    # Extract gene expression levels as tensors and move to GPU
    gene_data_tensors = [isolate_gene_values(adata, gene) for gene in gene_list]

    # Combine the spatial coordinates and gene data into a single tensor (on GPU)
    gene_data = torch.cat(gene_data_tensors, dim=1)
    
    # Shuffle data and sample if requested
    indices = torch.randperm(gene_data.size(0))  # Shuffle indices on GPU
    if head != 0:
        indices = indices[:head]
    
    coordinates = coordinates[indices]
    gene_data = gene_data[indices]
    
    # Preprocessing Step 1: Identify columns (genes) that sum to 0 and drop those columns
    column_sums = torch.sum(gene_data, dim=0)
    non_zero_columns = column_sums != 0  # Boolean mask for non-zero columns
    
    if not torch.all(non_zero_columns):
        # Get the names of the genes that are dropped (zero sum columns)
        dropped_genes = [gene_list[i] for i in range(len(gene_list)) if not non_zero_columns[i].item()]
        print(f"{dropped_genes} dropped from gene_list due to not being observed.")
    
    gene_data = gene_data[:, non_zero_columns]  # Keep only non-zero columns
    column_sums = column_sums[non_zero_columns]  # Adjust column sums for non-zero columns
    gene_list = [gene_list[i] for i in range(len(gene_list)) if non_zero_columns[i].item()]

    # Divide each column by its corresponding sum
    gene_data = gene_data / column_sums

    # Preprocessing Step 2: Scale 'x' and 'y', and log-transform gene expressions
    coordinates[:, 0] = coordinates[:, 0] / 5000 - 0.5948  # Scale x
    coordinates[:, 1] = coordinates[:, 1] / 5000 - 0.5389  # Scale y

    # Log-transform gene expressions: log(1+x)
    gene_data = torch.log1p(gene_data)

    # Preprocessing Step 3: Normalize gene data to have a standard deviation of 1
    std_devs = torch.std(gene_data, dim=0)
    if torch.any(std_devs == 0):
        zero_std_genes = torch.nonzero(std_devs == 0).flatten().tolist()
        print(f"error: {zero_std_genes} have zero standard deviation.")
    else:
        gene_data = gene_data / std_devs  # Normalize genes by their std deviation

    # Convert spatial coordinates to tensor X and gene expressions to tensor Y
    X = coordinates.to(device)
    Y = gene_data.to(device)

    return adata, X, Y, gene_list  # adata kept on CPU; X and Y on GPU
    
# rewritten 21 Oct to make all preprocessing on GPU
# def isolate_gene_values(adata, gene_name): # from .raw
#     return pd.DataFrame(adata.raw[:, gene_name].X.toarray(), columns=[gene_name], index=adata.obs_names)

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

#     # Preprocessing Step 2: Divide 'x' and 'y' by 5000 and then centre to (0,0); log(1+x) the genes
#     df['x'] = df['x'] / 5000 - 0.5948
#     df['y'] = df['y'] / 5000 - 0.5389
#     for gene in gene_list:
#         df[gene] = np.log(df[gene]+1)

#     # Preprocessing Step 3: Make all standard deviations equal to 1
#     for gene in gene_list:
#         std = df[gene].std()
#         if std == 0:
#             print(f"error: {gene} has zero standard deviation")
#         else:
#             df[gene] = df[gene] / std

#     # Output Objects
#     # Convert 'x' and 'y' columns into a tensor for spatial coordinates, X
#     X = torch.tensor(df[['x', 'y']].values, dtype=torch.float64).to(device)
#     # Convert gene columns into a tensor for gene expression values, Y
#     Y = torch.tensor(df[gene_list].values, dtype=torch.float64).to(device)
#     return adata, X, Y, df, gene_list

# Old load_data; replaced on 11 Oct 2024.
# def load_data(gene_list, head=0, puck_list = "all"):
#     # Load the H5AD file
#     file_path = os.path.expanduser('~/project/mouse_ovary_slide_seq_young_estrus.h5ad')
#     adata = sc.read_h5ad(file_path)

#     if puck_list != "all":
#         adata = adata[adata.obs['puck'].isin(puck_list)]
        
#     # Extract spatial coordinates
#     coordinates = pd.DataFrame(adata.obsm["spatial"], columns=['x', 'y'], index=adata.obs_names)
    
#     # Concatenate gene expression values with coordinates

#     # gene_list = [
#     #     "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1", "Cyp17a1"
#     # ]
    
#     # Collect gene values using the isolate_gene_values function
#     gene_data = [isolate_gene_values(adata, gene) for gene in gene_list]
    
#     # Combine coordinates with the gene data
#     df = pd.concat([coordinates] + gene_data, axis=1)

#     # shuffle
#     df = df.sample(frac=1)
#     # Shuffle and sample the DataFrame
#     if head>0:
#         df = df.head(head)
    
#     # Normalize 'x' and 'y' by their medians
#     df['x'] = df['x'] / 1000
#     df['y'] = df['y'] / 1000

#     ### preprocessing
#     # # normalize specific gene columns
#     # for gene in gene_list:
#     #     if df[gene].mean()!=0:
#     #         df[gene] = df[gene]/df[gene].mean()
            
#     # for gene in gene_list:
#     #     df[gene] = np.log(df[gene]+1)
#     ### preprocessing
    
#     # Convert 'x' and 'y' columns into a tensor for spatial coordinates
#     X = torch.tensor(df[['x', 'y']].values, dtype=torch.float64)
    
#     # Define the list of genes for Y tensor (you can modify this list as needed)
#     # Convert gene columns into a tensor for gene expression values
#     Y = torch.tensor(df[gene_list].values, dtype=torch.float64)
#     return adata, X, Y, df


# def load_real_data(head=15000): # Obsolete
#     # Load the H5AD file
#     adata = sc.read_h5ad('ovary_Puck_230517_39.h5ad')
#     coordinates = pd.DataFrame(adata.obsm["spatial"], columns=['x', 'y'], index=adata.obs_names)
#     df = pd.concat([
#         coordinates,
#         isolate_gene_values(adata, "Serpine2"),
#         isolate_gene_values(adata, "Tagln"),
#         isolate_gene_values(adata, "Acta2"),
#         isolate_gene_values(adata, "Mgp"),
#         isolate_gene_values(adata, "S100a6"),
#         isolate_gene_values(adata, "Col1a2"),
#         isolate_gene_values(adata, "Nr5a2"),
#         isolate_gene_values(adata, "Inhba"),
#         isolate_gene_values(adata, "Tpm2"),
#         isolate_gene_values(adata, "Tdrd5")
#     ], axis=1)
#     df = df.sample(frac=1)
#     df = df.head(head)
    
#     df['x'] = df['x'] / df['x'].median()
#     df['y'] = df['y'] / df['y'].median()
    
#     # Normalize all gene columns by dividing by 1000
#     genes_to_include = ["Serpine2", "Tagln", "Acta2", "Mgp", "S100a6", "Col1a2", "Nr5a2", "Inhba", "Tpm2", "Tdrd5"]
#     for gene in genes_to_include:
#         df[gene] = df[gene] / 1000
#     X = torch.tensor(df[['x', 'y']].values, dtype=torch.float64)

#     # List of gene columns
#     genes_to_include = ["Serpine2", "Tagln", "Acta2", "Mgp", "S100a6", "Col1a2", "Nr5a2", "Inhba", "Tpm2", "Tdrd5"]
#     # Convert the gene columns into a tensor
#     Y = torch.tensor(df[genes_to_include].values, dtype=torch.float64)
#     return X,Y

def load_synthetic_data (number_of_locations):
    X = simulate_locations(number_of_locations, dims).to(device)
    # Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma = random_search_parameters(p,X)
    # alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma) 
    Y = simulate_gp_data(X, compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)).detach().to(device)
    return X,Y, Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha_matrix, nu_matrix, sigma_matrix

# Produce p plots so that I can visualise how these data look like
def plot_gp_data(X, Y):
    """
    Plots the p-variate data for each variable.
    
    Parameters:
    - X (torch.Tensor): The matrix of locations of shape (n_locations, dimensions).
    - Y (torch.Tensor): The simulated dataset of shape (n_locations, p).
    """
    p = Y.size(1)
    
    # Create a plot for each variable
    for i in range(p):
        plt.figure(figsize=(8, 6))
        if X.size(1) == 2:  # 2D locations
            plt.scatter(X[:, 0].detach().numpy(), X[:, 1].detach().numpy(), c=Y[:, i].detach().numpy(), cmap='viridis', s=1)
            plt.colorbar(label=f'Variable {i+1}')
            plt.xlabel('X1')
            plt.ylabel('X2')
        elif X.size(1) == 1:  # 1D locations
            plt.plot(X.detach().numpy(), Y[:, i].detach().numpy(), '-o')
            plt.xlabel('X')
            plt.ylabel(f'Variable {i+1}')
        
        plt.title(f'Visualization of Variable {i+1}')
        plt.tight_layout()
        plt.show()
    return True

# def random_search_parameters(p, X, max_iterations=100000):
#     """
#     Randomly searches for a combination of parameters that satisfies a given clause.
    
#     Parameters:
#     - p (int): The length of the list W (number of parameters W1 to Wp).
#     - clause_function (callable): A function that takes (Delta_A, Delta_B, rho_A, rho_B, rho_V, W) as inputs
#                                   and returns True if the clause is satisfied, False otherwise.
#     - max_iterations (int): The maximum number of random samples to test.

#     Returns:
#     - A tuple of PyTorch tensors if a solution is found.
#     - None if no solution is found within the max_iterations.
#     """
#     for _ in range(max_iterations):
#         # Generate random values for the parameters as PyTorch tensors
#         Delta_A = torch.tensor(random.uniform(torch.finfo(torch.float64).eps, 10), dtype=torch.float64).to(device)
#         Delta_B = torch.tensor(random.uniform(torch.finfo(torch.float64).eps, 10), dtype=torch.float64).to(device)
#         rho_A = torch.tensor(random.uniform(-.99, 0.99), dtype=torch.float64).to(device)
#         rho_B = torch.tensor(random.uniform(-.99, 0.99), dtype=torch.float64).to(device)
#         rho_V = torch.tensor(random.uniform(-.99, 0.99), dtype=torch.float64).to(device)
#         W = torch.tensor([random.uniform(torch.finfo(torch.float64).eps, 10) for _ in range(p)], dtype=torch.float64).to(device)
#         alpha = torch.tensor([random.uniform(torch.finfo(torch.float64).eps, 10) for _ in range(p)], dtype=torch.float64).to(device)
#         nu = torch.tensor([random.uniform(torch.finfo(torch.float64).eps, 10) for _ in range(p)], dtype=torch.float64).to(device)
#         sigma = torch.tensor([random.uniform(-10, 10) for _ in range(p)], dtype=torch.float64).to(device)
        
        
#         alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
#         # Check if the clause is satisfied
#         K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
#         K = (K + K.mT)/2
#         if is_positive_definite(K):
#             return Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma
    
#     # If no solution is found, return None
#     return None

# Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma = random_search_parameters(p,X)
# alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)

# print("Alpha matrix:\n", alpha_matrix)
# print("Nu matrix:\n", nu_matrix)
# print("Sigma matrix:\n", sigma_matrix)

# # Compute the Matérn covariance matrix
# K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
# print("Matérn covariance matrix K:\n", K)

# Now simulate in pytorch number_of_locations locations

def is_positive_definite(matrix):
    """Check if a matrix is positive definite."""
    try:
        torch.linalg.cholesky(matrix)
        return True
    except RuntimeError:
        return False

# Define the custom autograd function for the Bessel function of the second kind
class BesselKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, x):
        # Store v and x for the backward pass
        ctx.save_for_backward(v, x)
        
        # Convert tensors to numpy for SciPy compatibility
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        
        # Compute the Bessel function using SciPy
        output = torch.tensor(kv(v_np, x_np), dtype=torch.float64)
        
        # Return the output as a tensor
        return output.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        v, x = ctx.saved_tensors
        
        # Convert tensors to numpy for gradient calculations
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        
        # Numerical derivative with respect to x
        epsilon_x = 1e-12
        grad_x = (kv(v_np, x_np + epsilon_x) - kv(v_np, x_np - epsilon_x)) / (2 * epsilon_x)
        grad_x = torch.tensor(grad_x, dtype=torch.float64).to(x.device)
        
        # Derivative with respect to v (using kvp, which gives the derivative of kv with respect to v)
        grad_v = torch.tensor(kvp(v_np, x_np), dtype=torch.float64).to(v.device)
        
        # Multiply by the incoming gradient (chain rule)
        grad_input_x = grad_output * grad_x
        grad_input_v = grad_output * grad_v
        
        return grad_input_v, grad_input_x

def matern_kernel(pairwise_distances, length_scale, nu, sigma, epsilon=1e-9):
    """
    Computes the Matérn covariance matrix with support for broadcasting over nu, length_scale, and sigma2.

    Parameters:
    - pairwise_distances (torch.Tensor): Pairwise distances, shape (n_locations, n_locations) or broadcasted to (p, p, n_locations, n_locations).
    - length_scale (torch.Tensor): Length scale parameter, can be broadcasted (e.g., shape (p, p, 1, 1)).
    - nu (torch.Tensor): Smoothness parameter, can be broadcasted (e.g., shape (p, p, 1, 1)).
    - sigma2 (torch.Tensor): Variance parameter, can be broadcasted (e.g., shape (p, p, 1, 1)).
    - epsilon (float): A small perturbation to ensure nu != 0.5.

    Returns:
    - covariance_matrix (torch.Tensor): The computed Matérn covariance matrix, with shape depending on input broadcasting.
    """
    # Only move to the device if tensors are not already on the correct device
    if pairwise_distances.device != device:
        pairwise_distances = pairwise_distances.to(device)
    if nu.device != device:
        nu = nu.to(device)
    if length_scale.device != device:
        length_scale = length_scale.to(device)
    if sigma.device != device:
        sigma = sigma.to(device)
    sigma2=sigma**2

    # Add a tiny perturbation to nu if it's exactly 0.5
    nu = torch.where(nu == 0.5, nu + epsilon, nu)

    # Compute the scaled distances using broadcasting
    scaled_distances = torch.sqrt(2 * nu) * (pairwise_distances / length_scale)
    
    # Clamp the values of scaled_distances to avoid extreme numbers
    scaled_distances = torch.clamp(scaled_distances, min=1e-10, max=1e6).to(device)

    # Use the custom Bessel function with autograd
    bessel_term = BesselKFunction.apply(nu, scaled_distances)
    scaling_term = (2 ** (1.0 - nu)) / torch.exp(torch.lgamma(nu))
    covariance_matrix = sigma2 * scaling_term * (scaled_distances ** nu) * bessel_term
    
    # Set diagonal elements where pairwise_distances == 0 to sigma2
    covariance_matrix = torch.where(pairwise_distances == 0, sigma2, covariance_matrix)
    
    return covariance_matrix

# zeroth order approximation of matern function w.r.t. distance input.  This makes sense because the matern kernel is smooth w.r.t. the distance.
def approx_matern_kernel_marginal(X, alpha_i, nu_i, sigma_i, epsilon=1e-9, number_of_distances=500):
    # Step 1: Compute pairwise distances in X using pdist (no gradients needed for distances)
    pairwise_distances_condensed = torch.pdist(X, p=2).detach()  # Detach to avoid gradient tracking
    
    # Step 2: Precompute the Matérn kernel for a range of distances
    max_dist = pairwise_distances_condensed.max().detach()
    min_dist = pairwise_distances_condensed.min().detach()
    
    distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=X.device)
    
    # Create a dictionary to store precomputed kernel values (no gradients required for distances_grid)
    kernel_dict = torch.empty(number_of_distances, dtype=torch.float64, device=X.device)
    
    # Compute the kernel for each distance and store it in the tensor
    for i, dist in enumerate(distances_grid):
        kernel_dict[i] = matern_kernel(dist, nu_i, alpha_i, sigma_i)
    
    # Step 3: Normalize distances and map them to precomputed values using rounding (no gradients needed here)
    # Normalize the pairwise distances to fall between 0 and number_of_distances-1
    normalized_distances = ((pairwise_distances_condensed - min_dist) / (max_dist - min_dist)).detach()
    indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()
    
    # Step 4: Create an empty full matrix to store the kernel values
    n = X.size(0)  # Number of points
    K = torch.zeros((n, n), dtype=torch.float64, device=X.device)

    # Get the upper triangular indices without the diagonal (needed for pdist indexing)
    triu_indices = torch.triu_indices(n, n, offset=1, device=X.device)

    # Fill in the upper triangular part of the matrix using precomputed kernel values
    K[triu_indices[0], triu_indices[1]] = kernel_dict[indices]

    # Symmetrize the matrix by copying the upper triangular part to the lower triangular part
    K = K + K.T

    # Step 5: Manually fill the diagonal with sigma_i**2 (preserving gradient tracking for sigma_i)
    diag_indices = torch.arange(n, device=X.device)
    K[diag_indices, diag_indices] = sigma_i**2

    # Add a small epsilon value to the diagonal for numerical stability
    K += torch.eye(n, dtype=torch.float64, device=X.device) * epsilon
    return K


# replaced 25 Oct 2024 to take full advantage of the symmetric nature of K.  Using pdist instead of cdist
# also, it is not clear to me that this handles the diagonal case correctly. 
def approx_matern_kernel_marginal_old(X, alpha_i, nu_i, sigma_i, epsilon=1e-9,number_of_distances=500):    
    # Step 1: Compute pairwise distances in X_batch (no gradients needed for distances)
    pairwise_distances = torch.cdist(X, X).detach()  # Detach to avoid gradient tracking

    # Mask the diagonal to ignore zeros
    pairwise_distances_no_diag = pairwise_distances[~torch.eye(pairwise_distances.size(0), dtype=bool, device=X.device)]

    # Step 2: Precompute the Matérn kernel for a range of distances
    max_dist = torch.max(pairwise_distances_no_diag).detach()
    min_dist = torch.min(pairwise_distances_no_diag).detach()
    distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=device)
    
    # Create a dictionary to store precomputed kernel values (no gradients required for distances_grid)
    kernel_dict = torch.empty(number_of_distances, dtype=torch.float64, device=device)
    
    # Compute the kernel for each distance and store it in the tensor
    for i, dist in enumerate(distances_grid):
        kernel_dict[i] = matern_kernel(dist, alpha_i, nu_i, sigma_i)
    
    # Step 3: Normalize distances and map them to precomputed values using rounding (no gradients needed here)
    # Normalize the pairwise distances to fall between 0 and number_of_distances-1
    normalized_distances = ((pairwise_distances - min_dist) / (max_dist - min_dist)).detach()  # Detach here too
    indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()
    
    # Ensure that indices have no negative values by replacing negative indices with -1
    # Set the diagonal indices to -1 to handle self-pairs
    indices = torch.where(indices < 0, torch.tensor(-1, device=device), indices).detach()
    diag_indices = torch.arange(pairwise_distances.size(0), device=X.device)
    indices[diag_indices, diag_indices] = -1  # Set diagonal entries to -1 explicitly
    
    # Step 4: Create K_approx by indexing into the precomputed kernel dictionary (no gradients tracked here)
    K = torch.empty_like(pairwise_distances, device=device)
    
    # Loop through the pairwise distances and fill in K_approx based on precomputed values
    for i in range(pairwise_distances.size(0)):
        for j in range(pairwise_distances.size(1)):
            index = indices[i, j].item()  # Get the integer index
            if index == -1:  # Handle the diagonal case
                K[i, j] = sigma_i**2
            else:
                K[i, j] = kernel_dict[index]
    K += torch.eye(K.size(0), device=K.device) * epsilon
    K = (K + K.mT) / 2
    return K

# Now that we have the alpha, nu, and sigma matrices, we want to define a function 
# Input are those three matrices, as well as X, the matrix of locations
# Output is the matern covariance matrix, computed by the function 
# matern_kernel(pairwise_distances, nu, length_scale, sigma2)
def compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X):
    """
    Computes the Matérn covariance matrix using the provided alpha, nu, and sigma matrices.

    Parameters:
    - alpha_matrix (torch.Tensor): The alpha matrix of shape (p, p).
    - nu_matrix (torch.Tensor): The nu matrix of shape (p, p).
    - sigma_matrix (torch.Tensor): The sigma matrix of shape (p, p).
    - X (torch.Tensor): The matrix of locations of shape (n_locations, dimensions).

    Returns:
    - K (torch.Tensor): The Matérn covariance matrix of shape (n_locations * p, n_locations * p).
    """
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    # Compute pairwise distances between locations (n_locations, n_locations)
    pairwise_distances = torch.cdist(X, X)
    
    # Expand pairwise distances to (p, p, n_locations, n_locations) for broadcasting
    pairwise_distances_expanded = pairwise_distances.unsqueeze(0).unsqueeze(0).expand(p, p, n_locations, n_locations)
    
    # Ensure the correct shapes for broadcasting
    alpha_expanded = alpha_matrix.unsqueeze(-1).unsqueeze(-1)  # Shape: (p, p, 1, 1)
    nu_expanded = nu_matrix.unsqueeze(-1).unsqueeze(-1)  # Shape: (p, p, 1, 1)
    sigma_expanded = sigma_matrix.unsqueeze(-1).unsqueeze(-1)  # Shape: (p, p, 1, 1)

    # Compute the covariance matrix using broadcasting
    K_blocks = matern_kernel(
        pairwise_distances_expanded, 
        nu_expanded,  
        alpha_expanded,  
        sigma_expanded
    )
    
    # Reshape to create the block covariance matrix (n_locations * p, n_locations * p)
    K = K_blocks.permute(0, 2, 1, 3).reshape(p * n_locations, p * n_locations)
    return K

def approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X, epsilon=1e-9, number_of_distances=500):
    """
    Approximates the Matérn covariance matrix using precomputed values with a single set of locations X.

    Parameters:
    - X (torch.Tensor): Matrix of shape (n_locations, dimensions) representing the locations.
    - alpha_matrix (torch.Tensor): Matrix of shape (p, p) for the alpha parameter of the Matérn kernel.
    - nu_matrix (torch.Tensor): Matrix of shape (p, p) for the nu parameter of the Matérn kernel.
    - sigma_matrix (torch.Tensor): Matrix of shape (p, p) for the sigma parameter of the Matérn kernel.
    - epsilon (float): Small constant to add to the diagonal to ensure positive definiteness.
    - number_of_distances (int): Number of precomputed distances to use in quantization.

    Returns:
    - K_approx (torch.Tensor): Approximate Matérn covariance matrix of shape (n_locations * p, n_locations * p).
    """
    device = X.device
    n_locations = X.size(0)
    p = alpha_matrix.size(0)

    # Step 1: Compute pairwise distances using pdist and detach to avoid gradient tracking
    pairwise_distances_condensed = torch.pdist(X, p=2).detach()

    # Step 2: Precompute the Matérn kernel for a range of distances
    max_dist = pairwise_distances_condensed.max().detach()
    min_dist = pairwise_distances_condensed.min().detach()

    distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=device)
    kernel_dict = torch.empty((p, p, number_of_distances), dtype=torch.float64, device=device)

    for i in range(p):
        for j in range(p):
            for k, dist in enumerate(distances_grid):
                kernel_dict[i, j, k] = matern_kernel(dist, nu_matrix[i, j], alpha_matrix[i, j], sigma_matrix[i, j])

    # Step 3: Normalize distances and map them to precomputed kernel values using rounding
    normalized_distances = ((pairwise_distances_condensed - min_dist) / (max_dist - min_dist)).detach()
    indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()

    # Step 4: Create the block covariance matrix (p, p, n_locations, n_locations)
    K_blocks = torch.zeros((p, p, n_locations, n_locations), dtype=torch.float64, device=device)
    
    # Get the upper triangular indices for the block matrices
    triu_indices = torch.triu_indices(n_locations, n_locations, 1, device=device)

    # Fill the upper triangular part of K_blocks
    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        index = indices[idx].item()
        if index == -1:
            K_blocks[:, :, i, j] = sigma_matrix**2
        else:
            K_blocks[:, :, i, j] = kernel_dict[:, :, index]
    
    # Symmetrize the block matrices by adding the transpose
    K_blocks += K_blocks.transpose(2, 3)

    # Fill the diagonal with sigma_matrix**2
    for i in range(n_locations):
        K_blocks[:, :, i, i] = sigma_matrix**2

    # Step 5: Add epsilon to the diagonal to ensure positive definiteness
    K_blocks[:, :, range(n_locations), range(n_locations)] += epsilon

    # Step 6: Reshape the block matrix to (n_locations * p, n_locations * p)
    K_approx = K_blocks.permute(0, 2, 1, 3).reshape(p * n_locations, p * n_locations)

    return K_approx

# replaced 25 Oct 2024 to use pdist not cdist so as to take advantage of the symmetric nature of K
# def approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X, epsilon=1e-9, number_of_distances=500):
#     """
#     Approximates the Matérn covariance matrix using precomputed values with a single set of locations X.

#     Parameters:
#     - X (torch.Tensor): Matrix of shape (n_locations, dimensions) representing the locations.
#     - nu_matrix (torch.Tensor): Matrix of shape (p, p) for the nu parameter of the Matérn kernel.
#     - alpha_matrix (torch.Tensor): Matrix of shape (p, p) for the alpha parameter of the Matérn kernel.
#     - sigma_matrix (torch.Tensor): Matrix of shape (p, p) for the sigma parameter of the Matérn kernel.
#     - epsilon (float): Small constant to add to the diagonal to ensure positive definiteness.
#     - number_of_distances (int): Number of precomputed distances to use in quantization.

#     Returns:
#     - K_approx (torch.Tensor): Approximate Matérn covariance matrix of shape (n_locations * p, n_locations * p).
#     """
#     device = X.device
#     n_locations = X.size(0)
#     p = alpha_matrix.size(0)
    
#     # Step 1: Compute pairwise distances (detached) between the locations X
#     pairwise_distances = torch.cdist(X, X).detach()

#     # Step 2: Precompute the Matérn kernel for a range of distances
#     max_dist = torch.flatten(pairwise_distances).max().detach()
#     min_dist = torch.flatten(pairwise_distances).min().detach()

#     distances_grid = torch.linspace(min_dist, max_dist, number_of_distances, dtype=torch.float64, device=device)
#     kernel_dict = torch.empty((p, p, number_of_distances), dtype=torch.float64, device=device)

#     for i in range(p):
#         for j in range(p):
#             for k, dist in enumerate(distances_grid):
#                 kernel_dict[i, j, k] = matern_kernel(dist, nu_matrix[i, j], alpha_matrix[i, j], sigma_matrix[i, j])

#     # Step 3: Normalize distances and map them to precomputed kernel values using rounding
#     normalized_distances = ((pairwise_distances - min_dist) / (max_dist - min_dist)).detach()
#     indices = (normalized_distances * (number_of_distances - 1)).round().long().detach()
    
#     # Step 4: Create the block covariance matrix (p, p, n_locations, n_locations)
#     K_blocks = torch.empty((p, p, n_locations, n_locations), dtype=torch.float64, device=device)
    
#     for i in range(n_locations):
#         for j in range(n_locations):
#             index = indices[i, j].item()
#             if index == -1:
#                 K_blocks[:, :, i, j] = sigma_matrix**2
#             else:
#                 K_blocks[:, :, i, j] = kernel_dict[:, :, index]

#     # Step 5: Reshape the block matrix to (n_locations * p, n_locations * p)
#     K_approx = K_blocks.permute(0, 2, 1, 3).reshape(p * n_locations, p * n_locations)

#     return K_approx

def compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma):
    """
    Computes p by p matrices alpha, nu, and sigma based on the provided parameters.

    Parameters:
    - Delta_A, Delta_B, rho_A, rho_B, rho_V (torch.float64): Scalars.
    - W, alpha, nu, sigma (torch.float64): 1D tensors of size p.

    Returns:
    - alpha_matrix, nu_matrix, sigma_matrix: p x p matrices of computed values.
    """
    p = W.size(0)
    # print("p in compute_parameter_matrices is", p)
    dim = 2  # Given constant value

    # Calculate alpha_ij matrix
    alpha_i_squared = (alpha.unsqueeze(1)**2).to(device)  # shape: (p, 1)
    alpha_j_squared = (alpha.unsqueeze(0)**2).to(device)  # shape: (1, p)
    alpha_matrix = (torch.sqrt((alpha_i_squared + alpha_j_squared) / 2 + Delta_B * (1 - rho_B))).to(device)
    
    # Calculate nu_ij matrix
    nu_matrix = ((nu.unsqueeze(1) + nu.unsqueeze(0)) / 2 + Delta_A * (1 - rho_A)).to(device)
    
    # Calculate sigma_ij matrix
    W_i = (W.unsqueeze(1)).to(device)# shape: (p, 1)
    W_j = (W.unsqueeze(0)).to(device) # shape: (1, p)
    sigma_matrix = (
        W_i * W_j * rho_V * alpha_matrix ** (-2 * Delta_A - (nu.unsqueeze(0) + nu.unsqueeze(1))) *
        torch.exp(
            torch.lgamma((nu.unsqueeze(0) + nu.unsqueeze(1)) / 2 + dim / 2) +
            torch.lgamma(nu_matrix) -
            torch.lgamma(nu_matrix + dim / 2)
        )
    ).to(device)  
     
    # Set diagonal entries for alpha_matrix with grad tracking
    alpha_matrix = alpha_matrix + torch.diag(alpha - torch.diag(alpha_matrix)).to(device)
    # Set diagonal entries for nu_matrix with grad tracking
    nu_matrix = nu_matrix + torch.diag(nu - torch.diag(nu_matrix)).to(device)
    # Set diagonal entries for sigma_matrix with grad tracking
    sigma_matrix = sigma_matrix + torch.diag(sigma - torch.diag(sigma_matrix)).to(device)

    return alpha_matrix, nu_matrix, sigma_matrix

# Example usage
# Delta_A = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
# Delta_B = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
# rho_A = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
# rho_B = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
# rho_V = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)

# W = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
# alpha = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)
# nu = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64, requires_grad=True)
# sigma = torch.tensor([0.1, 0.2 , 0.3], dtype=torch.float64, requires_grad=True)

def simulate_locations(number_of_locations, dimensions=2, range_min=-3.0, range_max=3.0):
    """
    Simulates random locations within a specified range.

    Parameters:
    - number_of_locations (int): The number of locations to simulate.
    - dimensions (int): The number of dimensions for each location (e.g., 2 for 2D, 3 for 3D).
    - range_min (float): The minimum value for the location coordinates.
    - range_max (float): The maximum value for the location coordinates.

    Returns:
    - locations (torch.Tensor): A tensor of shape (number_of_locations, dimensions) containing the simulated locations.
    """
    # Generate random locations within the specified range
    locations = torch.FloatTensor(number_of_locations, dimensions).uniform_(range_min, range_max)
    return locations


# Now that we have X and K,  simulate p-variate data sampled at locations in X with a Gaussian process along Normal(0,K) 
def simulate_gp_data(X, K):
    """
    Simulates a p-variate dataset sampled at locations in X with a Gaussian process.
    
    Parameters:
    - X (torch.Tensor): The matrix of locations of shape (n_locations, dimensions).
    - K (torch.Tensor): The covariance matrix computed using the Matérn kernel of shape (n_locations * p, n_locations * p).

    Returns:
    - Y (torch.Tensor): The simulated dataset of shape (n_locations, p).
    """
    # Get the number of locations and the dimensionality of the data
    n_locations = X.size(0)
    p = K.size(0) // n_locations

    # Sample from a multivariate normal distribution with mean 0 and covariance K
    mean = torch.zeros(K.size(0), dtype=torch.float64)


    # ####    ####    ####    ####    ####
    # try:
    #     L = torch.linalg.cholesky(K)
    #     print("Covariance matrix is positive definite.")
    # except RuntimeError:
    #     print("Covariance matrix is not positive definite.")
        
    # # Perform eigendecomposition
    # eigenvalues, eigenvectors = torch.linalg.eig(K)
    
    # # Separate the real and imaginary parts (if necessary)
    # eigenvalues_real = eigenvalues.real
    
    # # Sort the eigenvalues and the corresponding eigenvectors
    # sorted_indices = torch.argsort(eigenvalues_real)
    # sorted_eigenvalues = eigenvalues_real[sorted_indices]
    # print("Smallest Eigenvalue:")
    # print(sorted_eigenvalues[0])
    # print("Largest Eigenvalue:")
    # print(sorted_eigenvalues[-1])
    # print(K)

    # ####    ####    ####    ####    ####    
    K = (K + K.mT)/2
    Y = torch.distributions.MultivariateNormal(mean, covariance_matrix=K).rsample()
    # Reshape the output to have shape (n_locations, p)
    Y = Y.reshape(n_locations, p)
    return Y

# The particular setting in Genton's paper, with p=3
def Genton_parametrisation(number_of_locations, dims):    
    true_nu1_value, true_a1_value, true_sigma_1_value, true_nu2_value, true_a2_value, \
    true_sigma_2_value, true_nu3_value, true_a3_value, true_sigma_3_value, true_nu12_value, \
    true_a12_value, true_sigma_12_value, true_nu13_value, true_a13_value, true_sigma_13_value, \
    true_nu23_value, true_a23_value, true_sigma_23_value = [
        1.2, 0.01, 1.0, 0.6, 0.02, 1.0, 0.3, 0.03, 1.0, 1.093, 0.0205, -0.286, 
        1.092, 0.0263, -0.181, 0.990, 0.0282, 0.274
    ]
    alpha_matrix = torch.tensor([
        [true_a1_value,  true_a12_value, true_a13_value],
        [true_a12_value, true_a2_value,  true_a23_value],
        [true_a13_value, true_a23_value, true_a3_value]
    ])
    
    nu_matrix = torch.tensor([
        [true_nu1_value, true_nu12_value, true_nu13_value],
        [true_nu12_value, true_nu2_value, true_nu23_value],
        [true_nu13_value, true_nu23_value, true_nu3_value]
    ])
    
    sigma_matrix = torch.tensor([
        [true_sigma_1_value,  true_sigma_12_value, true_sigma_13_value],
        [true_sigma_12_value, true_sigma_2_value,  true_sigma_23_value],
        [true_sigma_13_value, true_sigma_23_value, true_sigma_3_value]
    ])
    X = simulate_locations(number_of_locations, dims)
    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    K = (K + K.mT)/2
    # nugget = 0  # or another small positive value
    # K += nugget * torch.eye(K.shape[0], device=K.device)
    Y = simulate_gp_data(X, K).detach()
    return X,Y,K, alpha_matrix, nu_matrix, sigma_matrix
    
def store_as_df(alpha_matrix, nu_matrix, sigma_matrix):
    data_dict={}
    for i in range(alpha_matrix.size(0)):
        for j in range(alpha_matrix.size(1)):
            if i<=j:
                data_dict[f'alpha_matrix_{i+1}{j+1}'] = alpha_matrix[i, j].item()
                data_dict[f'nu_matrix_{i+1}{j+1}'] = nu_matrix[i, j].item()
                data_dict[f'sigma_matrix_{i+1}{j+1}'] = sigma_matrix[i, j].item()
    # Create a DataFrame to store the values
    df = pd.DataFrame([data_dict])
    return df

# Compute the negative log-likelihood loss
def negative_log_likelihood(y, cov_matrix):
    n = y.shape[0]
    L = cholesky(cov_matrix, upper=False)
    # alpha, _ = torch.triangular_solve(y.reshape(-1, 1), L, upper=False) # the function is depreciated
    alpha = torch.linalg.solve_triangular(L, y.reshape(-1, 1), upper=False) # don't use ".view" always use ".reshape"
    log_likelihood = 0.5 * torch.sum(alpha ** 2)
    log_likelihood += torch.sum(torch.log(torch.diag(L)))
    log_likelihood += 0.5 * n * torch.log(torch.tensor(2 * torch.pi, device=y.device, dtype=y.dtype))
    return log_likelihood

def psd_condition_checker(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch):
    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
    K += torch.eye(K.size(0))*1e-8
    K = (K + K.mT)/2
    if not is_positive_definite(K):
        print("PSD condition failed.")
        # Perform eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        eigenvalues_real = eigenvalues.real
        # Sort the eigenvalues and the corresponding eigenvectors
        sorted_indices = torch.argsort(eigenvalues_real)
        sorted_eigenvalues = eigenvalues_real[sorted_indices]
        print("Smallest Eigenvalue:")
        print(sorted_eigenvalues[0])
        
    return is_positive_definite(K)
