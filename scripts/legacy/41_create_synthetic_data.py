# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/41_create_synthetic_data.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Goal of this notebook is to create synthetic data that look like Hattie's data.
#
# 1. Training locations ($X_{train}$) and values ($Y_{train}$) (size 15000), * 3*30 copies
# 2. Testing locations ($X_{test}$) (size 900), * 3*30 copies
# 3. The covariance matrix ($K_{test}$) at those locations, which are stored in 3*30 csv's, too.
# 4. Theoretically, the ground truth parameters could be different for these notebooks, just like there are many different genes. 

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
os.makedirs(os.path.expanduser("~/project/41_1_train_data/"), exist_ok=True)
os.makedirs(os.path.expanduser("~/project/41_2_test_locations/"), exist_ok=True)
os.makedirs(os.path.expanduser("~/project/41_3_test_cov/"), exist_ok=True)

# %%
X *= 200

# %%
def inhomogeneous_parametrisation_fixed_locations(X):    
    alpha_values = [1, 0.2, 0.03]
    nu_values = [2.4, 0.6, 0.15]
    sigma_values = [1, 1, 1]
    optimized_marginal_params = [
        torch.tensor([alpha, nu, sigma], dtype=torch.float64)
        for alpha, nu, sigma in zip(alpha_values, nu_values, sigma_values)
    ]
    # Extract the optimized alpha, nu, and sigma from the list
    alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    
    p = 3
    Delta_A = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    Delta_B = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_A = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_B = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_V = torch.tensor(-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    W = torch.full((p,), torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    
    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    
    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    K = (K + K.mT)/2
    # print(is_positive_definite(K))
    nugget = 1e-10  # or another small positive value
    K += nugget * torch.eye(K.shape[0], device=K.device)
    Y = simulate_gp_data(X, K).detach()
    return Y,K
# print(inhomogeneous_parametrisation_fixed_locations(X[0:2]))

# %%
for i in range(1):
    print(f"Creating dataset {i}...")
    # Shuffle X and perform train-test split
    X_shuffled = X[torch.randperm(X.size(0))]
    X_train = X_shuffled[:15000]
    X_test = X_shuffled[-1000:]
    
    # Generate Y_train and K_test
    Y_train, K_train = Genton_parametrisation_fixed_locations(X_train)
    Y_test, K_test = Genton_parametrisation_fixed_locations(X_test)
    
    # Split Y_train and K_test into individual parts
    Y_train_0, Y_train_1, Y_train_2 = Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]
    K_test_0, K_test_1, K_test_2 = K_test[:1000, :1000], K_test[1000:2000, 1000:2000], K_test[2000:3000, 2000:3000]
    
    # Convert to pandas DataFrames for saving as CSV
    pd.DataFrame(X_train.numpy()).to_csv(f"~/project/41_1_train_data/X_train_{i}.csv", index=False, header=False)
    pd.DataFrame(X_test.numpy()).to_csv(f"~/project/41_2_test_locations/X_test_{i}.csv", index=False, header=False)

    # Save each Y_train column and each K_test block as CSV
    for j, Y_train_j in enumerate([Y_train_0, Y_train_1, Y_train_2], start=0):
        pd.DataFrame(Y_train_j.numpy()).to_csv(f"~/project/41_1_train_data/Y_train_{i}_{j}.csv", index=False, header=False)
        
    for j, K_test_j in enumerate([K_test_0, K_test_1, K_test_2], start=0):
        pd.DataFrame(K_test_j.numpy()).to_csv(f"~/project/41_3_test_cov/K_test_{i}_{j}.csv", index=False, header=False)
        
    # Optional: Clear variables to save memory if running in a memory-limited environment
    del K_train, Y_train, K_test, Y_test, K_test_1, K_test_2, K_test_0, Y_train_1, Y_train_2, Y_train_0

print("homogeneous dataset created")

# %%
i += 1
num_datasets = 1

while num_datasets > 0:
    # Define paths for checking existence
    x_train_path = os.path.expanduser(f"~/project/41_1_train_data/X_train_{i}.csv")
    x_test_path = os.path.expanduser(f"~/project/41_2_test_locations/X_test_{i}.csv")
    y_train_paths = [os.path.expanduser(f"~/project/41_1_train_data/Y_train_{i}_{j}.csv") for j in range(3)]
    k_test_paths = [os.path.expanduser(f"~/project/41_3_test_cov/K_test_{i}_{j}.csv") for j in range(3)]

    # # Check if any of the files for this `i` exist
    # if os.path.exists(x_train_path) or os.path.exists(x_test_path) or any(os.path.exists(path) for path in y_train_paths + k_test_paths):
    #     # If any file exists, increment i and skip to the next iteration
    #     i += 1
    #     continue

    # If no files exist, proceed to create dataset `i`
    print(f"Creating dataset {i}...")
    
    # Shuffle X and perform train-test split
    X_shuffled = X[torch.randperm(X.size(0))]
    X_train = X_shuffled[:15000]
    X_test = X_shuffled[-1000:]
    
    # Generate Y_train and K_test
    Y_train, K_train = inhomogeneous_parametrisation_fixed_locations(X_train)
    Y_test, K_test = inhomogeneous_parametrisation_fixed_locations(X_test)
    
    # Split Y_train and K_test into individual parts
    Y_train_0, Y_train_1, Y_train_2 = Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]
    K_test_0, K_test_1, K_test_2 = K_test[:1000, :1000], K_test[1000:2000, 1000:2000], K_test[2000:3000, 2000:3000]
    
    # Convert to pandas DataFrames for saving as CSV
    pd.DataFrame(X_train.numpy()).to_csv(f"~/project/41_1_train_data/X_train_{i}.csv", index=False, header=False)
    pd.DataFrame(X_test.numpy()).to_csv(f"~/project/41_2_test_locations/X_test_{i}.csv", index=False, header=False)

    # Save each Y_train column and each K_test block as CSV
    for j, Y_train_j in enumerate([Y_train_0, Y_train_1, Y_train_2], start=0):
        pd.DataFrame(Y_train_j.numpy()).to_csv(f"~/project/41_1_train_data/Y_train_{i}_{j}.csv", index=False, header=False)
        
    for j, K_test_j in enumerate([K_test_0, K_test_1, K_test_2], start=0):
        pd.DataFrame(K_test_j.detach().numpy()).to_csv(f"~/project/41_3_test_cov/K_test_{i}_{j}.csv", index=False, header=False)
        
    # Optional: Clear variables to save memory if running in a memory-limited environment
    del K_train, Y_train, K_test, Y_test, K_test_1, K_test_2, K_test_0, Y_train_1, Y_train_2, Y_train_0

    # Decrement the count of datasets to create
    num_datasets -= 1

    # Move to the next `i` value
    i += 1

print("inhomogeneous dataset created")

# %%
# ## At this point, we have created our desired $X_{train}, Y_{train}$ for training, $X_{test}, K_{test}$ for validation.  We have also stored them to 41_1 (training data), 41_2 (testing locations), and 41_3 (K_test)

# %%
# # inhomogeneous_params

# %%
alpha_values = [1, 0.2, 0.03]
nu_values = [2.4, 0.6, 0.15]
sigma_values = [1, 1, 1]
optimized_marginal_params = [
    torch.tensor([alpha, nu, sigma], dtype=torch.float64)
    for alpha, nu, sigma in zip(alpha_values, nu_values, sigma_values)
]
# Extract the optimized alpha, nu, and sigma from the list
alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)

p = 3
Delta_A = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
Delta_B = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
rho_A = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
rho_B = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
rho_V = torch.tensor(-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
W = torch.full((p,), torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)

print("alpha_matrix\n",alpha_matrix.detach().numpy())
print("\nnu_matrix\n",nu_matrix.detach().numpy())
print("\nsigma_matrix\n",sigma_matrix.detach().numpy())

# %%
pass

