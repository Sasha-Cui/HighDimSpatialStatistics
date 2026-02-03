##fitting_functions
# 25 Oct 2024: made sure everything is in the order of alpha, nu, sigma.

# The functions defined herein are used for fitting marginal and cross parameters in the matern model.  
# This block contains marginal and cross optimisation codes that adaptively halves the stepsizes.
torch.autograd.set_detect_anomaly(False)

def optimize_marginal_parameters(X, Y, number_of_groups,number_of_cycles = 20, steps_per_batch=1, print_early_stopping_epochs = False):
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

    def printer_function(alpha_i, nu_i, sigma_i):
        # print(f"{nu_i.item():.3f}, {alpha_i.item():.3f}, {sigma_i.item():.3f}, {nu_i.grad.item():.3f}, {alpha_i.grad.item():.3f}, {sigma_i.grad.item():.3f}")
        print(f"the values are {alpha_i.item():.16f}, {nu_i.item():.3f}, {sigma_i.item():.3f}")
        print(f"the gradients are {alpha_i.grad.item():.3f}, {nu_i.grad.item():.16f}, {sigma_i.grad.item():.3f}")
        return None
    
    p = Y.size(1)
    n_locations = X.size(0)
    
    # Calculate the size of each group
    group_size = n_locations // number_of_groups
    # Split X and Y into smaller chunks
    X_groups = torch.split(X, group_size)
    Y_groups = torch.split(Y, group_size)
    
    optimized_params = []

    # Define different sets of initial learning rates and initialization values
    learning_rates = [
        {'alpha_lr': 0.00005, 'nu_lr': 0.001, 'sigma_lr': 0.01},
        {'alpha_lr': 0.0003, 'nu_lr': 0.008, 'sigma_lr': 0.25},
        {'alpha_lr': 0.0001, 'nu_lr': 0.005, 'sigma_lr': 0.1}
    ]
    initializations = [
        {'alpha_init': 0.005, 'nu_init': 1.0, 'sigma_init': 1.0},
        {'alpha_init': 0.01, 'nu_init': 1.0, 'sigma_init': 1.0},
        {'alpha_init': 0.05, 'nu_init': 1.0, 'sigma_init': 1.0},
        {'alpha_init': 0.1, 'nu_init': 1.0, 'sigma_init': 1.0},
        {'alpha_init': 0.2, 'nu_init': 1.0, 'sigma_init': 1.0}
    ]

    for lr_set, init_set in itertools.product(learning_rates, initializations):
        for i in range(p):
            loss_history = []  # To store the loss for each epoch
            
            # # Initialize alpha_i, nu_i, and sigma_i with requires_grad=True for optimization
            # alpha_i = torch.tensor(0.1, dtype=torch.float64).to(device).requires_grad_(True)
            # nu_i = torch.tensor(0.9, dtype=torch.float64).to(device).requires_grad_(True)
            # sigma_i = torch.tensor(1.0, dtype=torch.float64).to(device).requires_grad_(True)

            # Initialize alpha_i, nu_i, and sigma_i with requires_grad=True for optimization
            alpha_i = torch.tensor(init_set['alpha_init'], dtype=torch.float64, device=device).requires_grad_(True)
            nu_i = torch.tensor(init_set['nu_init'], dtype=torch.float64, device=device).requires_grad_(True)
            sigma_i = torch.tensor(init_set['sigma_init'], dtype=torch.float64, device=device).requires_grad_(True)
    
            # Early stopping parameters
            tolerance = 1e-15  # Threshold for considering convergence
            patience = 10  # Number of epochs with no improvement to wait before stopping
            best_loss = float('inf')
            best_params = {'alpha': None, 'nu': None, 'sigma': None}
            epochs_no_improve = 0
            times_lr_reduced = 0 # reduce the lr by 3 times at most
            
            # Define the optimizer
            # optimizer = optim.Adam([
            #     {'params': alpha_i, 'lr': 0.00005},  # Learning rate for alpha_i
            #     {'params': nu_i, 'lr': 0.001},        # Learning rate for nu_i
            #     {'params': sigma_i, 'lr': 0.01}     # Learning rate for sigma_i
            # ])
            # optimizer = torch.optim.Adagrad([
            #     {'params': alpha_i, 'lr': 0.0003},  # Learning rate for alpha_i
            #     {'params': nu_i, 'lr': 0.008},       # Learning rate for nu_i
            #     {'params': sigma_i, 'lr': 0.25}      # Learning rate for sigma_i
            # ], lr_decay=0, weight_decay=0, eps=1e-15)

            optimizer = torch.optim.Adagrad([
            {'params': alpha_i, 'lr': lr_set['alpha_lr']},  # Learning rate for alpha_i
            {'params': nu_i, 'lr': lr_set['nu_lr']},        # Learning rate for nu_i
            {'params': sigma_i, 'lr': lr_set['sigma_lr']}   # Learning rate for sigma_i
            ], lr_decay=0, weight_decay=0, eps=1e-15)
    
            # Optimization loop
            for epoch in range(number_of_cycles):  # Number of cycles / epochs

                # 28 Oct attempts to do warm-up start
                # Warmup strategy: Linear warmup over 5 epochs
                if epoch <= 4:
                    warmup_factor = (epoch + 1) / 5  # Adjust the factor based on the epoch
                    for param_group, init_lr in zip(optimizer.param_groups, initial_lrs):
                        param_group['lr'] = init_lr * warmup_factor

                total_nll = 0
                try:
                    for X_batch, Y_batch in zip(X_groups, Y_groups):
                        # X_batch = X_batch.to(device)
                        # Y_batch = Y_batch.to(device)
                        for _ in range(steps_per_batch):                        
                            # Compute the covariance matrix K
                            K = approx_matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
                            ## 10 Oct 2024 replaced the following two lines with the line above
                            ## K = matern_kernel(torch.cdist(X_batch, X_batch), alpha_i, nu_i, sigma_i)
                            ## K += torch.eye(K.size(0), device=device) * 1e-9
                            
                            # Compute the NLL for the batch
                            nll = negative_log_likelihood(Y_batch[:, i], K) 
                            total_nll += nll.item()
                            
                            # Backpropagation
                            optimizer.zero_grad()
                            nll.backward()
    
                            # printer_function(alpha_i, nu_i, sigma_i) ### log
                            # 
                            # Optimization step
                            optimizer.step()
    
                            with torch.no_grad():
                                alpha_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)
                                nu_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)
                                sigma_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)
                    
                    # Store the total loss for this epoch
                    loss_history.append(total_nll)
                    
                    # Check for convergence for early stopping
                    if total_nll < best_loss - tolerance:
                        best_loss = total_nll
                        best_params = {'alpha': alpha_i.detach().clone(), 'nu': nu_i.detach().clone(), 'sigma': sigma_i.detach().clone()}

                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    
                    if epochs_no_improve >= patience:          
                        # Multiply learning rate by 0.1
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.1
                            times_lr_reduced += 1
                            
                        if times_lr_reduced>=3:
                            break
                            
                        if print_early_stopping_epochs:
                            print("Reducing learning rate by 0.1 at epoch", epoch)
                        
                        # Reset epochs_no_improve to continue training
                        epochs_no_improve = 0
                        
                # updated 26 Oct 2024 to adaptively reduce the learning rate in case of plateauing.
                # if epochs_no_improve >= patience:
                    #     if print_early_stopping_epochs:
                    #         print("marginal optimisation early stopping at epoch", epoch)
                    #     break
                    
                except Exception as e:
                    # Report the parameters that led to the error
                    print(f"Error encountered during epoch {epoch}: {e}")
                    print(f"Parameters that caused the error -> alpha_i: {alpha_i}, nu_i: {nu_i}, sigma_i: {sigma_i}")
                    
                    # Optionally, break or continue
                    
                    break  # Stop the loop if you want to halt on error
            
            # Plot the loss over epochs
            plt.plot(loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f"Loss over Epochs (alpha_lr={lr_set['alpha_lr']}, nu_lr={lr_set['nu_lr']}, sigma_lr={lr_set['sigma_lr']}, alpha_init={init_set['alpha_init']})")
            # plt.title(f"Loss over Epochs (alpha_lr={lr_set['alpha_lr']}, nu_lr={lr_set['nu_lr']}, sigma_lr={lr_set['sigma_lr']}, alpha_init={init_set['alpha_init']}, nu_init={init_set['nu_init']}, sigma_init={init_set['sigma_init']})")
            plt.show()
        
            # Store the optimized parameters
            optimized_params.append((alpha_i.item(), nu_i.item(), sigma_i.item()))    
            # Print the best parameters after training
            print(f"Final parameters -> alpha_i: {alpha_i}, nu_i: {nu_i}, sigma_i: {sigma_i}")
            print(f"Best parameters found -> alpha_i: {best_params['alpha']}, nu_i: {best_params['nu']}, sigma_i: {best_params['sigma']}")

    return optimized_params

# 18 Oct 2024 includes the saving and loading of parameters over different times. 
# Save and Load functions for cross parameters
def save_checkpoint_cross(epoch, Delta_A, Delta_B, rho_A, rho_B, rho_V, W, optimizer, best_loss, filename='cross_checkpoint.pth'):
    checkpoint_data = {
        'epoch': epoch,
        'Delta_A': Delta_A.clone().detach(),  
        'Delta_B': Delta_B.clone().detach(),  
        'rho_A': rho_A.clone().detach(),      
        'rho_B': rho_B.clone().detach(),     
        'rho_V': rho_V.clone().detach(),      
        'W': W.clone().detach(),              
        'model_state_dict': {
            'Delta_A': Delta_A.clone().detach(),
            'Delta_B': Delta_B.clone().detach(),
            'rho_A': rho_A.clone().detach(),
            'rho_B': rho_B.clone().detach(),
            'rho_V': rho_V.clone().detach(),
            'W': W.clone().detach(),
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint_data, filename)

def load_checkpoint_cross(filename='cross_checkpoint.pth'):
    if os.path.exists(filename):
        checkpoint = torch.load(filename,weights_only=True)
        return checkpoint
    return None

# Cross Parameters Optimization
def optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles=500,steps_per_batch=20,print_early_stopping_epochs = False , checkpoint_interval=50, max_time_hours=24):
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
    start_time = time.time()
    max_time_seconds = max_time_hours * 3600
    
    # Split X and Y into smaller chunks
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
    Delta_A = torch.tensor(0.9, dtype=torch.float64, device=device, requires_grad=True)
    Delta_B = torch.tensor(0.9, dtype=torch.float64, device=device, requires_grad=True)
    rho_A = torch.tensor(0.1, dtype=torch.float64, device=device, requires_grad=True)
    rho_B = torch.tensor(0.1, dtype=torch.float64, device=device, requires_grad=True)
    rho_V = torch.tensor(-0.1, dtype=torch.float64, device=device, requires_grad=True)
    # W = (torch.ones(p, dtype=torch.float64, requires_grad=True) * torch.finfo(torch.float64).eps).clone().detach().requires_grad_(True).to(device)
    W = torch.full((p,), torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True) # apparently this is better
    
    # Extract the optimized alpha, nu, and sigma from the list
    alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    
    # Define the optimizer
    initial_lr = 0.1
    # initial_lr = 10
    optimizer = optim.Adam([Delta_A, Delta_B, rho_A, rho_B, rho_V, W], lr=initial_lr)
    lr_log = []
    halving_log = []

    # try to load the checkpoint
    checkpoint = load_checkpoint_cross()
    if checkpoint:
        current_epoch = checkpoint['epoch']
        Delta_A.data = checkpoint['Delta_A'].clone().detach().to(device)
        Delta_B.data = checkpoint['Delta_B'].clone().detach().to(device)
        rho_A.data = checkpoint['rho_A'].clone().detach().to(device)
        rho_B.data = checkpoint['rho_B'].clone().detach().to(device)
        rho_V.data = checkpoint['rho_V'].clone().detach().to(device)
        W.data = checkpoint['W'].clone().detach().to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
        print(f"Resuming cross fitting from epoch {current_epoch}")
        print(f"Loaded Delta_A: {Delta_A}")
        print(f"Loaded Delta_B: {Delta_B}")
        print(f"Loaded rho_A: {rho_A}")
        print(f"Loaded rho_B: {rho_B}")
        print(f"Loaded rho_V: {rho_V}")
        print(f"Loaded W: {W}")
        print(f"Loaded best_loss: {best_loss}")
    else:
        current_epoch = 0
        
    # Optimization loop
    for epoch in range(current_epoch, number_of_cycles): # Number of Cycles
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
                    # K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
                    K = approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X_batch)# 20 Oct 2024 updated approximated matern computation
                    
                    # Step 3: Add a small noise for numerical stability; put this code inside the kernel function.
                    # K += torch.eye(K.size(0), device = K.device)*1e-8
                    # K = (K + K.mT) / 2
    
                    # Step 4: Compute the NLL for the batch
                    # print(Y_batch.size())
                    # print(X_batch.size())
                    # print(K.size())
                    
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
        
        # save parameters regularly and also before the time is up
        elapsed_time = time.time() - start_time
        if epoch % checkpoint_interval == 0:
            # Save checkpoint at regular intervals but continue running
            print(f"Saving cross fitting checkpoint at epoch {epoch}")
            save_checkpoint_cross(epoch, Delta_A, Delta_B, rho_A, rho_B, rho_V, W, optimizer, best_loss)

        if elapsed_time >= max_time_seconds - 600:
            # Save checkpoint and exit before hitting the time limit
            print(f"Saving cross fitting checkpoint before timeout at epoch {epoch}")
            save_checkpoint_cross(epoch, Delta_A, Delta_B, rho_A, rho_B, rho_V, W, optimizer, best_loss)
            return compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
            
    # After optimization
    if halving_log!=[]:
        print("\nLearning Rate Halving Log:", halving_log)
        
    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    # K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    K = approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X) # 20 Oct 2024 updated approximated matern computation
    K += torch.eye(K.size(0), dtype=K.dtype, device=K.device) * 1e-8 #
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

# 18 Oct 2024 update: added the auto-saving functionality
# def optimize_cross_parameters(optimized_marginal_params,X,Y,number_of_groups,number_of_cycles=500,steps_per_batch=20,print_early_stopping_epochs = False ):
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
    
#     # Split X and Y into smaller chunks
#     P = Y.size(1) # not needed probably
#     p = Y.size(1)
#     n_locations = X.size(0)
#     group_size = n_locations // number_of_groups
#     X_groups = torch.split(X, group_size)
#     Y_groups = torch.split(Y, group_size)
    
#     # Early stopping parameters
#     tolerance = 1e-15 # Threshold for considering convergence
#     patience = 150  # Number of epochs with no improvement to wait before stopping
#     best_loss = float('inf')
#     epochs_no_improve = 0
    
#     # Initialize the parameters to be optimized
#     Delta_A = torch.tensor(0.9, dtype=torch.float64).to(device).requires_grad_(True)
#     Delta_B = torch.tensor(0.9, dtype=torch.float64).to(device).requires_grad_(True)
#     rho_A = torch.tensor(0.1, dtype=torch.float64).to(device).requires_grad_(True)
#     rho_B = torch.tensor(0.1, dtype=torch.float64).to(device).requires_grad_(True)
#     rho_V = torch.tensor(-0.1, dtype=torch.float64).to(device).requires_grad_(True)
#     W = (torch.ones(p, dtype=torch.float64, requires_grad=True) * torch.finfo(torch.float64).eps).clone().detach().requires_grad_(True).to(device)
    
#     # Extract the optimized alpha, nu, and sigma from the list
#     alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
#     nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
#     sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False).to(device)
    
#     # Define the optimizer
#     initial_lr = 0.001
#     optimizer = optim.Adam([Delta_A, Delta_B, rho_A, rho_B, rho_V, W], lr=initial_lr)
#     lr_log = []
#     halving_log = []
        
#     # Optimization loop
#     for epoch in range(number_of_cycles):  # Number of Cycles
#         # print("epoch ", epoch, "\n")
#         total_nll = 0
#         try:
#             for X_batch, Y_batch in zip(X_groups, Y_groups):
#                 X_batch=X_batch.to(device)
#                 Y_batch=Y_batch.to(device)
#                 # code to recover the initial learning rate
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = initial_lr
#                 for _ in range(steps_per_batch):
#                     # forward pass 
#                     # Step 1: Compute the parameter matrices
#                     alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    
#                     # Step 2: Compute the Matérn covariance matrix
#                     K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
                    
#                     # Step 3: Add a small noise for numerical stability
#                     K += torch.eye(K.size(0))*1e-8
#                     K = (K + K.mT) / 2
    
#                     # Step 4: Compute the NLL for the batch
#                     nll = negative_log_likelihood(Y_batch, K)
#                     total_nll += nll.item()
    
#                     # Backpropagation
#                     optimizer.zero_grad()
#                     nll.backward()
    
#                     # Save the current model state (before attempting the step) and the gradients
#                     model_state_before_step = {
#                         'Delta_A': Delta_A.clone(),
#                         'Delta_B': Delta_B.clone(),
#                         'rho_A': rho_A.clone(),
#                         'rho_B': rho_B.clone(),
#                         'rho_V': rho_V.clone(),
#                         'W': W.clone()
#                     }
                    
#                     # Save gradients as well
#                     grad_state_before_step = {
#                         'Delta_A_grad': Delta_A.grad.clone(),
#                         'Delta_B_grad': Delta_B.grad.clone(),
#                         'rho_A_grad': rho_A.grad.clone(),
#                         'rho_B_grad': rho_B.grad.clone(),
#                         'rho_V_grad': rho_V.grad.clone(),
#                         'W_grad': W.grad.clone()
#                     }
    
#                     halving_attempts = 0
#                     success = False
#                     # Keep halving the learning rate if the condition fails, up to 10 times
#                     while not success and halving_attempts < 10:
#                         # Try optimization step
#                         optimizer.step()
#                         # Projection to ensure rho_A, rho_B, and rho_V remain < 1, 0<W<1
#                         with torch.no_grad():
#                             rho_A.clamp_(min=torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
#                             rho_B.clamp_(min=torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
#                             rho_V.clamp_(min=-1 + torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
#                             W.clamp_(max=1 - torch.finfo(torch.float64).eps, min=torch.finfo(torch.float64).eps)
#                             Delta_A.clamp_(min=torch.finfo(torch.float64).eps)
#                             Delta_B.clamp_(min=torch.finfo(torch.float64).eps)
    
#                         # Check if the K is positive semidefinite
#                         if psd_condition_checker(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch):
#                             success = True
#                         else:
#                             # Condition failed, revert to the previous state and halve the learning rate
#                             print(f"Condition failed at epoch {epoch}, halving learning rate. Attempt {halving_attempts + 1}")
#                             halving_log.append(f"Epoch {epoch}: LR halved at attempt {halving_attempts + 1}")
    
#                             # Restore model to the previous state
#                             with torch.no_grad():
#                                 Delta_A.copy_(model_state_before_step['Delta_A'])
#                                 Delta_B.copy_(model_state_before_step['Delta_B'])
#                                 rho_A.copy_(model_state_before_step['rho_A'])
#                                 rho_B.copy_(model_state_before_step['rho_B'])
#                                 rho_V.copy_(model_state_before_step['rho_V'])
#                                 W.copy_(model_state_before_step['W'])
    
#                             # Restore gradients as well
#                             Delta_A.grad.copy_(grad_state_before_step['Delta_A_grad'])
#                             Delta_B.grad.copy_(grad_state_before_step['Delta_B_grad'])
#                             rho_A.grad.copy_(grad_state_before_step['rho_A_grad'])
#                             rho_B.grad.copy_(grad_state_before_step['rho_B_grad'])
#                             rho_V.grad.copy_(grad_state_before_step['rho_V_grad'])
#                             W.grad.copy_(grad_state_before_step['W_grad'])
    
#                             # Halve the learning rate manually
#                             for param_group in optimizer.param_groups:
#                                 param_group['lr'] /= 2
#                             halving_attempts += 1
#                     if success:
#                         if halving_attempts >1:
#                             print(f"epoch {epoch} step {_} succeeded with learning rate: {optimizer.param_groups[0]['lr']}")
#                     else:
#                         print(f"epoch {epoch} step {_} failed after 10 halving attempts.")
#                     # Log the learning rate for this epoch
#                     lr_log.append(optimizer.param_groups[0]['lr'])

                        
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
#             if print_early_stopping_epochs:
#                 print("cross terms optimisation early stoppping at epoch ", epoch)
#             break
    
#     # After optimization

#     if halving_log!=[]:
#         print("\nLearning Rate Halving Log:", halving_log)
        
#     alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
#     K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
#     K += torch.eye(K.size(0), dtype=K.dtype, device=K.device) * 1e-8 #
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
#         # print(sorted_eigenvalues[-1])
    
#     return alpha_matrix, nu_matrix, sigma_matrix



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

#     def printer_function(alpha_i, nu_i, sigma_i):
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
#         alpha_i = torch.tensor(0.01, dtype=torch.float64).to(device).requires_grad_(True)
#         nu_i = torch.tensor(1.0, dtype=torch.float64).to(device).requires_grad_(True)
#         sigma_i = torch.tensor(1.0, dtype=torch.float64).to(device).requires_grad_(True)
        
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
#                         K = matern_kernel(torch.cdist(X_batch, X_batch), alpha_i, nu_i, sigma_i)
#                         K += torch.eye(K.size(0), device=K.device) * 1e-5
                        
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
#                         #     printer_function(alpha_i, nu_i, sigma_i)
                        
#                         # Optimization step
#                         optimizer.step()
#                         # if epoch > 196:
#                         #     print('after optimizer.step()')
#                         #     printer_function(alpha_i, nu_i, sigma_i)
                        
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
#     Delta_A = torch.tensor(0.9, dtype=torch.float64).to(device).requires_grad_(True)
#     Delta_B = torch.tensor(0.9, dtype=torch.float64).to(device).requires_grad_(True)
#     rho_A = torch.tensor(0.1, dtype=torch.float64).to(device).requires_grad_(True)
#     rho_B = torch.tensor(0.1, dtype=torch.float64).to(device).requires_grad_(True)
#     rho_V = torch.tensor(-0.1, dtype=torch.float64).to(device).requires_grad_(True)
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
#                     K += torch.eye(X_batch.size(0) * p, device=K.device) * 5e-5
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