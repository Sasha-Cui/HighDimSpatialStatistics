##fitting_functions
# The functions defined herein are used for fitting marginal and cross parameters in the matern model.  
# This block contains marginal and cross optimisation codes that adaptively halves the stepsizes.
torch.autograd.set_detect_anomaly(False)

def optimize_marginal_parameters(X, Y, number_of_groups,number_of_cycles = 100, steps_per_batch=5, print_early_stopping_epochs = False):
    """
    Optimize the parameters alpha_i, nu_i, and sigma_i for each variable i by minimizing the NLL using batch learning.
    
    Parameters:
    - X (torch.Tensor): Locations matrix of shape (n_locations, dimensions).
    - Y (torch.Tensor): Simulated data of shape (n_locations, p).
    - number_of_groups (int): Number of groups to divide the dataset into for batch learning.
    - steps_per_batch (int): Number of optimization steps to perform on each batch before moving to the next one.
    
    Returns:
    - optimized_params (list): List of optimized (alpha, nu, sigma) for each variable.
    """

    def printer_function(nu_i, alpha_i, sigma_i):
        print(f"{nu_i.item():.3f}, {alpha_i.item():.3f}, {sigma_i.item():.3f}, {nu_i.grad.item():.3f}, {alpha_i.grad.item():.3f}, {sigma_i.grad.item():.3f}")
        return True
    
    p = Y.size(1)
    n_locations = X.size(0)
    
    # Calculate the size of each group
    group_size = n_locations // number_of_groups
    # Split X and Y into smaller chunks
    X_groups = torch.split(X, group_size)
    Y_groups = torch.split(Y, group_size)
    
    optimized_params = []
    
    for i in range(p):
        # Initialize alpha_i, nu_i, and sigma_i with requires_grad=True for optimization
        alpha_i = torch.tensor(0.01, dtype=torch.float64, requires_grad=True).to(device)
        nu_i = torch.tensor(1.0, dtype=torch.float64, requires_grad=True).to(device)
        sigma_i = torch.tensor(1.0, dtype=torch.float64, requires_grad=True).to(device)
        
        # Early stopping parameters
        tolerance = 1e-15  # Threshold for considering convergence
        patience = 150  # Number of epochs with no improvement to wait before stopping
        best_loss = float('inf')
        epochs_no_improve = 0
        
        # Define the optimizer
        initial_lr=0.1
        optimizer = optim.Adam([alpha_i, nu_i, sigma_i], lr=initial_lr)
        lr_log = []
        halving_log = []    
        
        # Optimization loop
        for epoch in range(number_of_cycles):  # Number of cycles / epochs
            total_nll = 0
            try:
                for X_batch, Y_batch in zip(X_groups, Y_groups):
                    # code to recover the initial learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = initial_lr
                    for _ in range(steps_per_batch):
                        
                        # Compute the covariance matrix K
                        K = matern_kernel(torch.cdist(X_batch, X_batch), nu_i, alpha_i, sigma_i)
                        K += torch.eye(K.size(0)) * 1e-10
                        
                        # Compute the NLL for the batch
                        nll = negative_log_likelihood(Y_batch[:, i], K) #???? what is i here?
                        total_nll += nll.item()
                        
                        # Backpropagation
                        optimizer.zero_grad()
                        nll.backward()

                        # Optimization step
                        optimizer.step()

                        with torch.no_grad():
                            nu_i.clamp_(min=torch.finfo(torch.float64).eps, max=10)
                            alpha_i.clamp_(min=torch.finfo(torch.float64).eps, max=10)
                            sigma_i.clamp_(min=torch.finfo(torch.float64).eps, max=10)
                
                # Check for convergence for early stopping
                if total_nll < best_loss - tolerance:
                    best_loss = total_nll
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if print_early_stopping_epochs:
                        print("marginal optimisation early stopping at epoch", epoch)
                    break
                    
            except Exception as e:
                # Report the parameters that led to the error
                print(f"Error encountered during epoch {epoch}: {e}")
                print(f"Parameters that caused the error -> nu_i: {nu_i}, alpha_i: {alpha_i}, sigma_i: {sigma_i}")
                
                # Optionally, break or continue
                break  # Stop the loop if you want to halt on error
                
        # Store the optimized parameters
        optimized_params.append((alpha_i.item(), nu_i.item(), sigma_i.item()))    
    return optimized_params

def optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles=500,steps_per_batch=20,print_early_stopping_epochs = False ):
    """
    Optimize the cross parameters using batch learning.
    
    Parameters:
    - optimized_marginal_params (list): List of optimized (alpha, nu, sigma) for each variable.
    - X (torch.Tensor): Locations matrix of shape (n_locations, dimensions).
    - Y (torch.Tensor): Simulated data of shape (n_locations, p).
    - number_of_groups (int): Number of groups to divide the dataset into for batch learning.
    - steps_per_batch (int): Number of optimization steps to perform on each batch before moving to the next one.
    
    Returns:
    - estimated_params_df (pd.DataFrame): DataFrame containing the estimated parameters.
    """
    
    # Split X and Y into smaller chunks
    P = Y.size(1) # not needed probably
    p = Y.size(1)
    n_locations = X.size(0)
    group_size = n_locations // number_of_groups
    X_groups = torch.split(X, group_size)
    Y_groups = torch.split(Y, group_size)
    
    # Early stopping parameters
    tolerance = 1e-15 # Threshold for considering convergence
    patience = 150  # Number of epochs with no improvement to wait before stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Initialize the parameters to be optimized
    Delta_A = torch.tensor(0.9, dtype=torch.float64, requires_grad=True).to(device)
    Delta_B = torch.tensor(0.9, dtype=torch.float64, requires_grad=True).to(device)
    rho_A = torch.tensor(0.1, dtype=torch.float64, requires_grad=True).to(device)
    rho_B = torch.tensor(0.1, dtype=torch.float64, requires_grad=True).to(device)
    rho_V = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True).to(device)
    W = (torch.ones(p, dtype=torch.float64, requires_grad=True) * torch.finfo(torch.float64).eps).clone().detach().requires_grad_(True).to(device)
    
    # Extract the optimized alpha, nu, and sigma from the list
    alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
    nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
    sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
    
    # Define the optimizer
    initial_lr = 0.001
    optimizer = optim.Adam([Delta_A, Delta_B, rho_A, rho_B, rho_V, W], lr=initial_lr)
    lr_log = []
    halving_log = []
        
    # Optimization loop
    for epoch in range(number_of_cycles):  # Number of Cycles
        # print("epoch ", epoch, "\n")
        total_nll = 0
        try:
            for X_batch, Y_batch in zip(X_groups, Y_groups):
                # code to recover the initial learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr
                for _ in range(steps_per_batch):
                    # forward pass 
                    # Step 1: Compute the parameter matrices
                    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    
                    # Step 2: Compute the Matérn covariance matrix
                    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
                    
                    # Step 3: Add a small noise for numerical stability
                    K += torch.eye(K.size(0))*1e-8
                    K = (K + K.mT) / 2
    
                    # Step 4: Compute the NLL for the batch
                    nll = negative_log_likelihood(Y_batch, K)
                    total_nll += nll.item()
    
                    # Backpropagation
                    optimizer.zero_grad()
                    nll.backward()
    
                    # Save the current model state (before attempting the step) and the gradients
                    model_state_before_step = {
                        'Delta_A': Delta_A.clone(),
                        'Delta_B': Delta_B.clone(),
                        'rho_A': rho_A.clone(),
                        'rho_B': rho_B.clone(),
                        'rho_V': rho_V.clone(),
                        'W': W.clone()
                    }
                    
                    # Save gradients as well
                    grad_state_before_step = {
                        'Delta_A_grad': Delta_A.grad.clone(),
                        'Delta_B_grad': Delta_B.grad.clone(),
                        'rho_A_grad': rho_A.grad.clone(),
                        'rho_B_grad': rho_B.grad.clone(),
                        'rho_V_grad': rho_V.grad.clone(),
                        'W_grad': W.grad.clone()
                    }
    
                    halving_attempts = 0
                    success = False
                    # Keep halving the learning rate if the condition fails, up to 10 times
                    while not success and halving_attempts < 10:
                        # Try optimization step
                        optimizer.step()
                        # Projection to ensure rho_A, rho_B, and rho_V remain < 1, 0<W<1
                        with torch.no_grad():
                            rho_A.clamp_(min=torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
                            rho_B.clamp_(min=torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
                            rho_V.clamp_(min=-1 + torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
                            W.clamp_(max=1 - torch.finfo(torch.float64).eps, min=torch.finfo(torch.float64).eps)
                            Delta_A.clamp_(min=torch.finfo(torch.float64).eps)
                            Delta_B.clamp_(min=torch.finfo(torch.float64).eps)
    
                        # Check if the K is positive semidefinite
                        if psd_condition_checker(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch):
                            success = True
                        else:
                            # Condition failed, revert to the previous state and halve the learning rate
                            print(f"Condition failed at epoch {epoch}, halving learning rate. Attempt {halving_attempts + 1}")
                            halving_log.append(f"Epoch {epoch}: LR halved at attempt {halving_attempts + 1}")
    
                            # Restore model to the previous state
                            with torch.no_grad():
                                Delta_A.copy_(model_state_before_step['Delta_A'])
                                Delta_B.copy_(model_state_before_step['Delta_B'])
                                rho_A.copy_(model_state_before_step['rho_A'])
                                rho_B.copy_(model_state_before_step['rho_B'])
                                rho_V.copy_(model_state_before_step['rho_V'])
                                W.copy_(model_state_before_step['W'])
    
                            # Restore gradients as well
                            Delta_A.grad.copy_(grad_state_before_step['Delta_A_grad'])
                            Delta_B.grad.copy_(grad_state_before_step['Delta_B_grad'])
                            rho_A.grad.copy_(grad_state_before_step['rho_A_grad'])
                            rho_B.grad.copy_(grad_state_before_step['rho_B_grad'])
                            rho_V.grad.copy_(grad_state_before_step['rho_V_grad'])
                            W.grad.copy_(grad_state_before_step['W_grad'])
    
                            # Halve the learning rate manually
                            for param_group in optimizer.param_groups:
                                param_group['lr'] /= 2
                            halving_attempts += 1
                    if success:
                        if halving_attempts >1:
                            print(f"epoch {epoch} step {_} succeeded with learning rate: {optimizer.param_groups[0]['lr']}")
                    else:
                        print(f"epoch {epoch} step {_} failed after 10 halving attempts.")
                    # Log the learning rate for this epoch
                    lr_log.append(optimizer.param_groups[0]['lr'])

                        
        except Exception as e:
            # Report the parameters that led to the error
            print(f"Error encountered during epoch {epoch}: {e}")
            print(f"Parameters that caused the error -> Delta_A: {Delta_A}, Delta_B: {Delta_B}, rho_A: {rho_A}, rho_B: {rho_B}, rho_V: {rho_V}, W: {W}")

            if not is_positive_definite(K):
                print("Warning: K is not positive definite.")
                # Perform eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eig(K)
                
                # Separate the real and imaginary parts (if necessary)
                eigenvalues_real = eigenvalues.real
                
                # Sort the eigenvalues and the corresponding eigenvectors
                sorted_indices = torch.argsort(eigenvalues_real)
                sorted_eigenvalues = eigenvalues_real[sorted_indices]
                print("Smallest Eigenvalue:", sorted_eigenvalues[0])
                # print("Largest Eigenvalue:")
                # print(sorted_eigenvalues[-1])
    
            # Stop the loop if you want to halt on error
            break
    
        # Check for convergence for early stopping
        if total_nll < best_loss - tolerance:
            best_loss = total_nll
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            if print_early_stopping_epochs:
                print("cross terms optimisation early stoppping at epoch ", epoch)
            break
    
    # After optimization

    if halving_log!=[]:
        print("\nLearning Rate Halving Log:", halving_log)
        
    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    K += torch.eye(K.size(0), dtype=K.dtype, device =K.device) * 1e-8 #
    K = (K + K.mT)/2
    if not is_positive_definite(K):
        print("Warning: K is not positive definite.")
        # Perform eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        
        # Separate the real and imaginary parts (if necessary)
        eigenvalues_real = eigenvalues.real
        
        # Sort the eigenvalues and the corresponding eigenvectors
        sorted_indices = torch.argsort(eigenvalues_real)
        sorted_eigenvalues = eigenvalues_real[sorted_indices]
        print("Smallest Eigenvalue:", sorted_eigenvalues[0])
        # print("Largest Eigenvalue:")
        # print(sorted_eigenvalues[-1])
    
    return alpha_matrix, nu_matrix, sigma_matrix



# This block contains marginal and cross optimisation codes that do not half the stepsizes.
# def optimize_marginal_parameters(X, Y, number_of_groups,number_of_cycles = 100, steps_per_batch=5):
#     """
#     Optimize the parameters alpha_i, nu_i, and sigma_i for each variable i by minimizing the NLL using batch learning.
    
#     Parameters:
#     - X (torch.Tensor): Locations matrix of shape (n_locations, dimensions).
#     - Y (torch.Tensor): Simulated data of shape (n_locations, p).
#     - number_of_groups (int): Number of groups to divide the dataset into for batch learning.
#     - steps_per_batch (int): Number of optimization steps to perform on each batch before moving to the next one.
    
#     Returns:
#     - optimized_params (list): List of optimized (alpha, nu, sigma) for each variable.
#     """

#     def printer_function(nu_i, alpha_i, sigma_i):
#         print(f"{nu_i.item():.3f}, {alpha_i.item():.3f}, {sigma_i.item():.3f}, {nu_i.grad.item():.3f}, {alpha_i.grad.item():.3f}, {sigma_i.grad.item():.3f}")
#         return True
        
#     p = Y.size(1)
#     n_locations = X.size(0)
    
#     # Calculate the size of each group
#     group_size = n_locations // number_of_groups
#     # Split X and Y into smaller chunks
#     X_groups = torch.split(X, group_size)
#     Y_groups = torch.split(Y, group_size)
    
#     optimized_params = []
    
#     for i in range(p):
#         # Initialize alpha_i, nu_i, and sigma_i with requires_grad=True for optimization
#         alpha_i = torch.tensor(0.01, dtype=torch.float64, requires_grad=True).to(device)
#         nu_i = torch.tensor(1.0, dtype=torch.float64, requires_grad=True).to(device)
#         sigma_i = torch.tensor(1.0, dtype=torch.float64, requires_grad=True).to(device)
        
#         # Define the optimizer
#         optimizer = optim.Adam([alpha_i, nu_i, sigma_i], lr=0.001)

#         # Early stopping parameters
#         tolerance = 1e-15  # Threshold for considering convergence
#         patience = 150  # Number of epochs with no improvement to wait before stopping
#         best_loss = float('inf')
#         epochs_no_improve = 0
        
#         # Optimization loop
#         for epoch in range(number_of_cycles):  # Number of cycles / epochs
#             # if epoch > 196:
#                 # print(f"epoch {epoch}")
#             total_nll = 0
#             try:
#                 for X_batch, Y_batch in zip(X_groups, Y_groups):
#                     for _ in range(steps_per_batch):
#                         optimizer.zero_grad()
#                         # Compute the covariance matrix K
#                         K = matern_kernel(torch.cdist(X_batch, X_batch), nu_i, alpha_i, sigma_i)
#                         K += torch.eye(K.size(0)) * 1e-5
                        
#                         # Compute the NLL for the batch
#                         nll = negative_log_likelihood(Y_batch[:, i], K)
#                         total_nll += nll.item()
                        
#                         # Backpropagation
#                         nll.backward()
                        
#                         # Gradient clipping
#                         # torch.nn.utils.clip_grad_norm_(nu_i, norm_type='inf', max_norm=100000.0)
#                         # torch.nn.utils.clip_grad_norm_(alpha_i, norm_type='inf', max_norm=100000.0)
#                         # torch.nn.utils.clip_grad_norm_(sigma_i, norm_type='inf', max_norm=100000.0)
                        
#                         # if epoch > 196:
#                         #     print('after clipping')
#                         #     printer_function(nu_i, alpha_i, sigma_i)
                        
#                         # Optimization step
#                         optimizer.step()
#                         # if epoch > 196:
#                         #     print('after optimizer.step()')
#                         #     printer_function(nu_i, alpha_i, sigma_i)
                        
#                         with torch.no_grad():
#                             nu_i.clamp_(min=torch.finfo(torch.float64).eps, max=10)
#                             alpha_i.clamp_(min=torch.finfo(torch.float64).eps, max=10)
#                             sigma_i.clamp_(min=torch.finfo(torch.float64).eps, max=10)
                
#                 # Check for convergence for early stopping
#                 if total_nll < best_loss - tolerance:
#                     best_loss = total_nll
#                     epochs_no_improve = 0
#                 else:
#                     epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     print("marginal optimisation early stopping at epoch", epoch)
#                     break
                    
#             except Exception as e:
#                 # Report the parameters that led to the error
#                 print(f"Error encountered during epoch {epoch}: {e}")
#                 print(f"Parameters that caused the error -> nu_i: {nu_i}, alpha_i: {alpha_i}, sigma_i: {sigma_i}")
                
#                 # Optionally, break or continue
#                 break  # Stop the loop if you want to halt on error
                
#         # Store the optimized parameters
#         optimized_params.append((alpha_i.item(), nu_i.item(), sigma_i.item()))    
#     return optimized_params

# def optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles=500,steps_per_batch=20):
#     """
#     Optimize the cross parameters using batch learning.
    
#     Parameters:
#     - optimized_marginal_params (list): List of optimized (alpha, nu, sigma) for each variable.
#     - X (torch.Tensor): Locations matrix of shape (n_locations, dimensions).
#     - Y (torch.Tensor): Simulated data of shape (n_locations, p).
#     - number_of_groups (int): Number of groups to divide the dataset into for batch learning.
#     - steps_per_batch (int): Number of optimization steps to perform on each batch before moving to the next one.
    
#     Returns:
#     - estimated_params_df (pd.DataFrame): DataFrame containing the estimated parameters.
#     """
#     p = Y.size(1)
#     n_locations = X.size(0)
    
#     # Calculate the size of each group
#     group_size = n_locations // number_of_groups
    
#     # Split X and Y into smaller chunks
#     X_groups = torch.split(X, group_size)
#     Y_groups = torch.split(Y, group_size)
    
#     # Early stopping parameters
#     tolerance = 1e-15 # Threshold for considering convergence
#     patience = 100  # Number of epochs with no improvement to wait before stopping
#     best_loss = float('inf')
#     epochs_no_improve = 0
    
#     # Initialize the parameters to be optimized
#     Delta_A = torch.tensor(0.9, dtype=torch.float64, requires_grad=True).to(device)
#     Delta_B = torch.tensor(0.9, dtype=torch.float64, requires_grad=True).to(device)
#     rho_A = torch.tensor(0.1, dtype=torch.float64, requires_grad=True).to(device)
#     rho_B = torch.tensor(0.1, dtype=torch.float64, requires_grad=True).to(device)
#     rho_V = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True).to(device)
#     W = (torch.ones(p, dtype=torch.float64, requires_grad=True) * torch.finfo(torch.float64).eps).clone().detach().requires_grad_(True).to(device)
    
#     # Extract the optimized alpha, nu, and sigma from the list
#     alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
#     nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
#     sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
    
#     # Define the optimizer
#     optimizer = optim.Adam([Delta_A, Delta_B, rho_A, rho_B, rho_V, W], lr=0.001)
    
#    # Optimization loop
#     for epoch in range(number_of_cycles):  # Number of Cycles
#         print("epoch ", epoch, "\n")
#         total_nll = 0
#         try:
#             for X_batch, Y_batch in zip(X_groups, Y_groups):
#                 for _ in range(steps_per_batch):
#                     optimizer.zero_grad()
                    
#                     # Step 1: Compute the parameter matrices
#                     alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
                    
#                     # Step 2: Compute the Matérn covariance matrix
#                     K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
                    
#                     # Step 3: Add a small noise for numerical stability
#                     K += torch.eye(X_batch.size(0) * p) * 5e-5
#                     K = (K + K.mT)/2
                    
#                     # Step 4: Compute the NLL for the batch
#                     nll = negative_log_likelihood(Y_batch, K)
#                     total_nll += nll.item()
                    
#                     # Backpropagation
#                     nll.backward()
                    
#                     # Gradient clipping # updated 16 Sept 2024 to individual gradiants clipped
#                     # torch.nn.utils.clip_grad_norm_(Delta_A, norm_type='inf', max_norm=5.0)
#                     # torch.nn.utils.clip_grad_norm_(Delta_B, norm_type='inf', max_norm=5.0)
#                     # torch.nn.utils.clip_grad_norm_(rho_A, norm_type='inf', max_norm=5.0)
#                     # torch.nn.utils.clip_grad_norm_(rho_B, norm_type='inf', max_norm=5.0)
#                     # torch.nn.utils.clip_grad_norm_(rho_V, norm_type='inf', max_norm=5.0)
#                     # torch.nn.utils.clip_grad_norm_(W, norm_type='inf', max_norm=5.0)
                    
#                     # Optimization step
#                     optimizer.step()
#                     # Printing tensors and their gradients with 3 significant figures
#                     print(f"Delta_A: {Delta_A.item():.3g}, Gradient: {Delta_A.grad.item():.3g}")
#                     print(f"Delta_B: {Delta_B.item():.3g}, Gradient: {Delta_B.grad.item():.3g}")
#                     print(f"rho_A: {rho_A.item():.3g}, Gradient: {rho_A.grad.item():.3g}")
#                     print(f"rho_B: {rho_B.item():.3g}, Gradient: {rho_B.grad.item():.3g}")
#                     print(f"rho_V: {rho_V.item():.3g}, Gradient: {rho_V.grad.item():.3g}")
#                     for i in range(W.size(0)):  # Loop through each element of W
#                         print(f"W[{i}]: {W[i].item():.3g}, Gradient: {W.grad[i].item():.3g}")
        
#                     # Projection to ensure rho_A, rho_B, and rho_V remain < 1, 0<W<1
#                     with torch.no_grad():
#                         rho_A.clamp_(min=torch.finfo(torch.float64).eps,max=1-torch.finfo(torch.float64).eps)
#                         rho_B.clamp_(min=torch.finfo(torch.float64).eps,max=1-torch.finfo(torch.float64).eps)
#                         rho_V.clamp_(min=-1+torch.finfo(torch.float64).eps,max=1-torch.finfo(torch.float64).eps)
#                         W.clamp_(max=1-torch.finfo(torch.float64).eps, min=torch.finfo(torch.float64).eps)
#                         Delta_A.clamp_(min=torch.finfo(torch.float64).eps)
#                         Delta_B.clamp_(min=torch.finfo(torch.float64).eps)

#                     # Printing tensors and their gradients with 3 significant figures
#                     print(f"Delta_A: {Delta_A.item():.3g}, Gradient: {Delta_A.grad.item():.3g}")
#                     print(f"Delta_B: {Delta_B.item():.3g}, Gradient: {Delta_B.grad.item():.3g}")
#                     print(f"rho_A: {rho_A.item():.3g}, Gradient: {rho_A.grad.item():.3g}")
#                     print(f"rho_B: {rho_B.item():.3g}, Gradient: {rho_B.grad.item():.3g}")
#                     print(f"rho_V: {rho_V.item():.3g}, Gradient: {rho_V.grad.item():.3g}")
#                     for i in range(W.size(0)):  # Loop through each element of W
#                         print(f"W[{i}]: {W[i].item():.3g}, Gradient: {W.grad[i].item():.3g}")
                        
#         except Exception as e:
#             # Report the parameters that led to the error
#             print(f"Error encountered during epoch {epoch}: {e}")
#             print(f"Parameters that caused the error -> Delta_A: {Delta_A}, Delta_B: {Delta_B}, rho_A: {rho_A}, rho_B: {rho_B}, rho_V: {rho_V}, W: {W}")

#             if not is_positive_definite(K):
#                 print("Warning: K is not positive definite.")
#                 # Perform eigendecomposition
#                 eigenvalues, eigenvectors = torch.linalg.eig(K)
                
#                 # Separate the real and imaginary parts (if necessary)
#                 eigenvalues_real = eigenvalues.real
                
#                 # Sort the eigenvalues and the corresponding eigenvectors
#                 sorted_indices = torch.argsort(eigenvalues_real)
#                 sorted_eigenvalues = eigenvalues_real[sorted_indices]
#                 print("Smallest Eigenvalue:", sorted_eigenvalues[0])
#                 # print("Largest Eigenvalue:")
#                 # print(sorted_eigenvalues[-1])
            
    
#             # Stop the loop if you want to halt on error
#             break
    
#         # Check for convergence for early stopping
#         if total_nll < best_loss - tolerance:
#             best_loss = total_nll
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print("cross terms optimisation early stoppping at epoch ", epoch)
#             break
    
#     # After optimization
#     alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    
#     K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
#     K = (K + K.mT)/2
#     if not is_positive_definite(K):
#         print("Warning: K is not positive definite.")
#         # Perform eigendecomposition
#         eigenvalues, eigenvectors = torch.linalg.eig(K)
        
#         # Separate the real and imaginary parts (if necessary)
#         eigenvalues_real = eigenvalues.real
        
#         # Sort the eigenvalues and the corresponding eigenvectors
#         sorted_indices = torch.argsort(eigenvalues_real)
#         sorted_eigenvalues = eigenvalues_real[sorted_indices]
#         print("Smallest Eigenvalue:", sorted_eigenvalues[0])
#         # print("Largest Eigenvalue:")
#         print(sorted_eigenvalues[-1])
    
#     return alpha_matrix, nu_matrix, sigma_matrix