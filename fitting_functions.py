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
    
    for i in range(p):
        loss_history = []  # To store the loss for each epoch
        
        # Initialize alpha_i, nu_i, and sigma_i with requires_grad=True for optimization
        alpha_i = torch.tensor(0.1, dtype=torch.float64).to(device).requires_grad_(True)
        nu_i = torch.tensor(0.9, dtype=torch.float64).to(device).requires_grad_(True)
        sigma_i = torch.tensor(1.0, dtype=torch.float64).to(device).requires_grad_(True)

        # Early stopping parameters
        tolerance = 1e-15  # Threshold for considering convergence
        patience = 25  # Number of epochs with no improvement to wait before stopping
        best_loss = float('inf')
        epochs_no_improve = 0
        times_lr_reduced = 0 # reduce the lr by 3 times at most
        
        # Define the optimizer
        # optimizer = optim.Adam([
        #     {'params': alpha_i, 'lr': 0.00005},  # Learning rate for alpha_i
        #     {'params': nu_i, 'lr': 0.001},        # Learning rate for nu_i
        #     {'params': sigma_i, 'lr': 0.01}     # Learning rate for sigma_i
        # ])

        optimizer = torch.optim.Adagrad([
            {'params': alpha_i, 'lr': 0.0003},  # Learning rate for alpha_i
            {'params': nu_i, 'lr': 0.008},       # Learning rate for nu_i
            {'params': sigma_i, 'lr': 0.25}      # Learning rate for sigma_i
        ], lr_decay=0, weight_decay=0, eps=1e-15)

        # Optimization loop
        for epoch in range(number_of_cycles):  # Number of cycles / epochs
            total_nll = 0
            try:
                for X_batch, Y_batch in zip(X_groups, Y_groups):
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
        plt.title('Loss over Epochs')
        plt.show()
    
        # Store the optimized parameters
        optimized_params.append((alpha_i.item(), nu_i.item(), sigma_i.item()))    
        print(alpha_i.item(), nu_i.item(), sigma_i.item())
    return optimized_params

def optimize_marginal_parameters_in_groups(lr_set, init_set, X_groups, Y_groups, number_of_cycles = 1, steps_per_batch = 1, print_early_stopping_epochs = False, sigma_is_known = True):
    """
    Optimize the parameters `alpha_i`, `nu_i`, and optionally `sigma_i` for each variable `i` 
    by minimizing the Negative Log Likelihood (NLL) using batch learning.

    Parameters:
    - lr_set (dict): Learning rate configuration with keys:
        - 'alpha_lr' (float): Learning rate for the `alpha` parameter.
        - 'nu_lr' (float): Learning rate for the `nu` parameter.
        - 'sigma_lr' (float, optional): Learning rate for the `sigma` parameter if `sigma_is_known=False`.
    - init_set (dict): Initialization values with keys:
        - 'alpha_init' (float): Initial value for the `alpha` parameter.
        - 'nu_init' (float): Initial value for the `nu` parameter.
        - 'sigma_init' (float, optional): Initial value for the `sigma` parameter if `sigma_is_known=False`.
    - X_groups (list of torch.Tensor): List of location matrices for each batch, each of shape `(batch_size, dimensions)`.
    - Y_groups (list of torch.Tensor): List of data matrices for each batch, each of shape `(batch_size, p)`.
    - number_of_cycles (int, optional): Number of optimization cycles (epochs) to run. Default is 1.
    - steps_per_batch (int, optional): Number of optimization steps per batch before moving to the next batch. Default is 1.
    - print_early_stopping_epochs (bool, optional): If True, prints the epochs when learning rates are reduced due to early stopping. Default is False.
    - sigma_is_known (bool, optional): If True, treats `sigma` as a known constant and excludes it from optimization. Default is True.

    Returns:
    - optimized_params (list of lists): Optimized parameters for each variable, where each entry contains:
        - `[alpha_i, nu_i]` if `sigma_is_known=True`.
        - `[alpha_i, nu_i, sigma_i]` if `sigma_is_known=False`.
    - best_params (list of dicts): Best parameters observed during training for each variable, with keys:
        - 'alpha' (torch.Tensor): Best value of `alpha_i`.
        - 'nu' (torch.Tensor): Best value of `nu_i`.
        - 'sigma' (torch.Tensor, optional): Best value of `sigma_i` if `sigma_is_known=False`.
    - loss_histories (list of lists): History of total NLL values for each variable across all epochs.

    Notes:
    - Early stopping is implemented to reduce the learning rate by a factor of 0.1 when the loss does not improve 
      for a specified number of epochs (`patience`). Training halts if the learning rate has been reduced twice 
      without further improvement.
    - Parameters are clamped during training to ensure numerical stability (e.g., `alpha`, `nu`, and `sigma` are kept positive).
    - Batch learning is used, where optimization is performed in groups of data points (`X_groups` and `Y_groups`).

    Raises:
    - Exception: If an error occurs during optimization, the function prints the parameters causing the issue and stops the process.

    Example Usage:
    ```python
    lr_set = {'alpha_lr': 0.01, 'nu_lr': 0.005, 'sigma_lr': 0.001}
    init_set = {'alpha_init': 1.0, 'nu_init': 0.5, 'sigma_init': 0.1}
    X_groups = [torch.randn(50, 2) for _ in range(10)]  # Example batches
    Y_groups = [torch.randn(50, 3) for _ in range(10)]  # Example data

    optimized_params, best_params, loss_histories = optimize_marginal_parameters_in_groups(
        lr_set, init_set, X_groups, Y_groups, number_of_cycles=10, steps_per_batch=2
    )
    ```
    """

    def printer_function(alpha_i, nu_i, sigma_i):
        # print(f"{nu_i.item():.3f}, {alpha_i.item():.3f}, {sigma_i.item():.3f}, {nu_i.grad.item():.3f}, {alpha_i.grad.item():.3f}, {sigma_i.grad.item():.3f}")
        print(f"the values are {alpha_i.item():.16f}, {nu_i.item():.3f}, {sigma_i.item():.3f}")
        if sigma_is_known:
            print(f"the gradients are {alpha_i.grad.item():.16f}, {nu_i.grad.item():.16f}")
        else:
            print(f"the gradients are {alpha_i.grad.item():.16f}, {nu_i.grad.item():.16f}, {sigma_i.grad.item():.16f}")
        return None
    
    p = Y_groups[0].size(1)
    n_locations = X_groups[0].size(0)
    optimized_params = []
    # Initialize a container to store loss history for each feature
    loss_histories = []
    best_params = []
    for i in range(p):
        loss_history = []  # To store the loss for each epoch of the current feature

        # Initialize alpha_i, nu_i, and sigma_i with requires_grad=True for optimization
        alpha_i = torch.tensor(init_set['alpha_init'], dtype=torch.float64, device=device).requires_grad_(True)
        nu_i = torch.tensor(init_set['nu_init'], dtype=torch.float64, device=device).requires_grad_(True)
        
        if sigma_is_known: 
            sigma_i = torch.tensor(1.0, dtype=torch.float64, device=device).requires_grad_(False)
            optimizer = torch.optim.Adagrad([
            {'params': alpha_i, 'lr': lr_set['alpha_lr']},  # Learning rate for alpha_i
            {'params': nu_i, 'lr': lr_set['nu_lr']}        # Learning rate for nu_i
            ], lr_decay=0, weight_decay=0, eps=1e-15)            
            best_param = {'alpha': None, 'nu': None, 'sigma': sigma_i}
        else:
            sigma_i = torch.tensor(init_set['sigma_init'], dtype=torch.float64, device=device).requires_grad_(True)
            optimizer = torch.optim.Adagrad([
            {'params': alpha_i, 'lr': lr_set['alpha_lr']},  # Learning rate for alpha_i
            {'params': nu_i, 'lr': lr_set['nu_lr']},        # Learning rate for nu_i
            {'params': sigma_i, 'lr': lr_set['sigma_lr']}   # Learning rate for sigma_i
            ], lr_decay=0, weight_decay=0, eps=1e-15)
            best_param = {'alpha': None, 'nu': None, 'sigma': None}
        
        # Early stopping parameters
        tolerance = 1e-15  # Threshold for considering convergence
        patience = 5  # Number of epochs with no improvement to wait before stopping
        best_loss = float('inf')
        epochs_no_improve = 0
        times_lr_reduced = 0 # reduce the lr by 2 times at most

        # Optimization loop
        for epoch in range(number_of_cycles):  # Number of cycles / epochs
            # 28 Oct attempts to do warm-up start
            # Warmup strategy: Linear warmup over 5 epochs
            # if epoch <= 4:
            #     warmup_factor = (epoch + 1) / 5  # Adjust the factor based on the epoch
            #     for param_group, init_lr in zip(optimizer.param_groups, [lr_set['alpha_lr'], lr_set['nu_lr'], lr_set['sigma_lr']]):
            #         param_group['lr'] = init_lr * warmup_factor
                    
            total_nll = 0
            try:
                
                for batch_idx, (X_batch, Y_batch) in enumerate(zip(X_groups, Y_groups)):
                    is_last_batch = (batch_idx == len(X_groups) - 1)# Determine if this is the last batch
                # 6 Nov 2024 replaced the following line with the 2 lines above
                # for X_batch, Y_batch in zip(X_groups, Y_groups):
                    # X_batch = X_batch.to(device)
                    # Y_batch = Y_batch.to(device)
                    for _ in range(steps_per_batch):                        

                        ## code to analyse why the matrices are not PSD.  Answer: approx_matern_kernel_marginal does not preserve it.    
                        # if marginal_approx_psd_condition_checker(X_batch, alpha_i, nu_i, sigma_i):
                        #     print("psd")
                        #     K = approx_matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)

                        #     # K = matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
                        # else:
                        #     print("not psd")
                        #     print(i,epoch)
                        #     break
                        
                        # if marginal_psd_condition_checker(X_batch, alpha_i, nu_i, sigma_i):
                        #     print("psd")
                        #     K = matern_kernel(torch.cdist(X_batch, X_batch), alpha_i, nu_i, sigma_i)
                        #     K += torch.eye(K.size(0), device=device) * 1e-9
                        #     # K = matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
                        # else:
                        #     print("not psd")
                        #     print(i,epoch)
                        #     break

                        # Compute the covariance matrix K
                        K = approx_matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
                        ## 10 Oct 2024 replaced the following two lines with the line above
                        # K = matern_kernel(torch.cdist(X_batch, X_batch), alpha_i, nu_i, sigma_i)
                        # K += torch.eye(K.size(0), device=device) * 1e-9
                        
                        # Compute the NLL for the batch
                        nll = negative_log_likelihood(Y_batch[:, i], K) 
                        
                        if is_last_batch: # Accumulate total_nll only for the last batch
                            total_nll += nll.item()
                        
                        # Backpropagation
                        optimizer.zero_grad()
                        nll.backward()

                        # printer_function(alpha_i, nu_i, sigma_i) ### log
                        # print(nll) ### log
                        # # Optimization step
                        optimizer.step()

                        with torch.no_grad():
                            alpha_i.clamp_(min=torch.finfo(torch.float64).eps, max=100)
                            nu_i.clamp_(min=torch.finfo(torch.float64).eps, max=100)
                            if not sigma_is_known:
                                sigma_i.clamp_(min=torch.finfo(torch.float64).eps, max=100)
                
                # Store the total loss for this epoch
                loss_history.append(total_nll)
                
                # Check for convergence for early stopping
                if total_nll < best_loss - tolerance:
                    best_loss = total_nll
                    if sigma_is_known:
                        best_param = {'alpha': alpha_i.detach().clone(), 'nu': nu_i.detach().clone(), 'sigma': sigma_i if sigma_is_known else sigma_i.detach().clone()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:          
                    # Multiply learning rate by 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1
                        times_lr_reduced += 1
                    if times_lr_reduced>=2:
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
                break  # Stop the loop if you want to halt on error
        
        # Store the loss history, best params, and optimized params for the current feature
        loss_histories.append(loss_history)
        best_params.append([best_param])
        optimized_params.append([alpha_i, nu_i, sigma_i])    
        # Print the best parameters after training
        print(f"Final param-> alpha_i: {alpha_i}, nu_i: {nu_i}, sigma_i: {sigma_i}")
        print(f"Best param -> alpha_i: {best_param['alpha']}, nu_i: {best_param['nu']}, sigma_i: {best_param['sigma']}")
    return optimized_params, best_params, loss_histories


# 10 Nov 2024, optimize in groups. 
def optimize_cross_parameters_in_groups(optimized_marginal_params, lr_set, X_groups, Y_groups, number_of_cycles=300,steps_per_batch=2, print_early_stopping_epochs = False, checkpoint_interval=50, max_time_hours=24, logging=False):
    """
    Optimize cross-covariance parameters using batch learning with group-wise updates.

    Parameters:
    - optimized_marginal_params (list of torch.Tensor): A list of tensors, each containing optimized marginal parameters 
      (alpha, nu, sigma) for each variable. These are used to initialize the cross-covariance computation.
    - lr_set (dict of torch.Tensor): A dictionary containing learning rates for parameters, with keys as parameter names 
      (e.g., 'Delta_A', 'Delta_B', etc.) and values as tensors providing learning rates for each parameter.
    - X_groups (list of torch.Tensor): List of location matrices, where each element is a tensor of shape 
      (group_size, dimensions) representing a subset of locations for batch learning.
    - Y_groups (list of torch.Tensor): List of data matrices corresponding to `X_groups`. Each tensor has 
      shape (group_size, p), where `p` is the number of variables or features.
    - number_of_cycles (int, optional): The maximum number of optimization cycles (epochs) to run. Default is 300.
    - steps_per_batch (int, optional): The number of optimization steps to perform within each batch before moving 
      to the next. Default is 20.
    - print_early_stopping_epochs (bool, optional): If True, prints the epoch number when early stopping is 
      triggered. Default is False.
    - checkpoint_interval (int, optional): Interval at which to save optimization checkpoints, in terms of 
      number of epochs. Default is 50.
    - max_time_hours (float, optional): Maximum time allowed for optimization, in hours. The function will attempt to 
      save a checkpoint and exit before reaching this limit. Default is 24.
    - logging (bool, optional): If True, enables verbose output for logging key steps, such as loading checkpoints, 
      parameter values, and status of early stopping. Default is False.

    Returns:
    - final_params, a list which includes
        - Delta_A (torch.Tensor): Final optimized value for the Delta_A parameter.
        - Delta_B (torch.Tensor): Final optimized value for the Delta_B parameter.
        - rho_A (torch.Tensor): Final optimized value for the rho_A parameter.
        - rho_B (torch.Tensor): Final optimized value for the rho_B parameter.
        - rho_V (torch.Tensor): Final optimized value for the rho_V parameter.
        - W (torch.Tensor): Final optimized weight vector, typically of shape (p,).
    - best_params, a list which includes all of the above, but at the epoch which has the lowest loss. 
    
    Notes:
    - This function is designed to perform optimization on cross-covariance parameters, using batch learning 
      and group-wise updates for scalability.
    - Checkpoints are saved regularly based on the specified `checkpoint_interval` and before timeout to prevent 
      loss of progress.
    - The function also implements early stopping if no improvement in the negative log-likelihood (NLL) is 
      observed within the specified patience threshold.
    - Learning rates may be halved multiple times if necessary to meet positive semi-definiteness conditions for 
      the cross-covariance matrix.
    """
    
    # the main optimization step code
    def perform_optimization_step_with_halving(
        optimizer, params, param_names, model_state_before_step, grad_state_before_step,
        max_halving_attempts, epoch, step, halving_log, lr_log,
        Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch
    ):
        """
        Perform an optimization step with learning rate halving until success or max attempts reached.
    
        Returns:
        - success (bool): Whether the optimization step was successful.
        """
        success = False
        halving_attempts = 0
    
        while not success and halving_attempts < max_halving_attempts:
            # Try optimization step
            optimizer.step()
            
            # Clamp parameter values
            with torch.no_grad():
                eps = torch.finfo(torch.float64).eps
                params_min_max = {
                    rho_A: (eps, 1 - eps),
                    rho_B: (eps, 1 - eps),
                    rho_V: (-1 + eps, 1 - eps),
                    W: (eps, 1 - eps),
                    Delta_A: (eps, None),
                    Delta_B: (eps, None)
                }
                for param, (min_val, max_val) in params_min_max.items():
                    param.clamp_(min=min_val, max=max_val)
            
            # Check if the K is positive semidefinite
            if cross_psd_condition_checker(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch):
                success = True
            else:
                print(f"Condition failed at epoch {epoch}, halving learning rate. Attempt {halving_attempts + 1}")
                halving_log.append(f"Epoch {epoch}: LR halved at attempt {halving_attempts + 1}")
    
                # Restore model state and gradients to the previous step
                with torch.no_grad():
                    for name, param in zip(param_names, params):
                        param.copy_(model_state_before_step[name])
                        param.grad.copy_(grad_state_before_step[f"{name}_grad"])
                
                # Halve the learning rate manually
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2
                halving_attempts += 1
        # Log success or failure message
        if success:
            if halving_attempts > 1:
                print(f"Epoch {epoch} step {step} succeeded with learning rate: {optimizer.param_groups[0]['lr']}")
        else:
            print(f"Epoch {epoch} step {step} failed after {max_halving_attempts} halving attempts.")
        # Log the learning rate for this epoch
        lr_log.append(optimizer.param_groups[0]['lr'])
        return success


    
    start_time = time.time()
    max_time_seconds = max_time_hours * 3600
    # Early stopping parameters
    tolerance = 1e-15 # Threshold for considering convergence
    patience = 5 # Number of epochs with no improvement to wait before stopping
    epochs_no_improve = 0
    best_loss = float('inf')
    best_params = {}
    loss_histories = []
    
    # Initialize the parameters to be optimized
    p = Y_groups[0].size(1)
    device = Y_groups[0].device
    Delta_A = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    Delta_B = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_A = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_B = torch.tensor(1-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_V = torch.tensor(-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    W = torch.full((p,), torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    # Extract the optimized alpha, nu, and sigma from the list
    alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)

    param_names = ['Delta_A', 'Delta_B', 'rho_A', 'rho_B', 'rho_V', 'W'] # Parameters to load
    params = [Delta_A, Delta_B, rho_A, rho_B, rho_V, W]
    
    # Initialize optimizer with specific learning rates from lr_set
    optimizer = optim.Adam(
        [{'params': param, 'lr': lr_set[name]} for name, param in zip(param_names, params)]
    )

    max_halving_attempts = 3
    lr_log = []
    halving_log = []

    # try to load the checkpoint
    checkpoint = load_checkpoint_cross()
    if checkpoint:
        # Restore checkpointed values
        current_epoch, best_loss = checkpoint['epoch'], checkpoint['best_loss']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for name, param in zip(param_names, params):
            param.data = checkpoint[name].clone().detach().to(device)
        if logging: # Logging
            print(f"Resuming cross fitting from epoch {current_epoch}; best loss thus far {best_loss}")
            for name, param in zip(param_names, params):
                print(f"Loaded {name}: {param}")
    else:
        current_epoch = 0
    
    # initialise best_params
    best_params = {name: param.clone() for name, param in zip(param_names, params)}
    # Optimization loop
    for epoch in range(current_epoch, number_of_cycles): # Number of Cycles
        total_nll = 0
        try:
            for batch_idx, (X_batch, Y_batch) in enumerate(zip(X_groups, Y_groups)):
                is_last_batch = (batch_idx == len(X_groups) - 1)# Determine if this is the last batch
                
                # code to recover the initial learning rate
                for param_group, name in zip(optimizer.param_groups, lr_set.keys()):
                    param_group['lr'] = lr_set[name]                
                
                for step in range(steps_per_batch):
                    # forward pass 
                    # Step 1: Compute the parameter matrices
                    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
                    
                    # Step 2: Compute the Matérn covariance matrix
                    K = approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X_batch)# 20 Oct 2024 updated approximated matern computation
                    
                    # Step 3: Compute the NLL for the batch
                    nll = negative_log_likelihood(Y_batch, K)
                    if is_last_batch: # Accumulate total_nll only for the last batch
                            total_nll += nll.item()
    
                    # Backpropagation
                    optimizer.zero_grad()
                    nll.backward()
    
                    # Save the current model state and gradients, before attempting the updating step
                    model_state_before_step = {name: param.clone() for name, param in zip(param_names, params)}
                    grad_state_before_step = {f"{name}_grad": param.grad.clone() for name, param in zip(param_names, params)}
                    
                    # Keep halving the learning rate if the condition fails, up to max_halving_attempts times
                    success = perform_optimization_step_with_halving(
                        optimizer, params, param_names, model_state_before_step, grad_state_before_step,
                        max_halving_attempts, epoch, step, halving_log, lr_log,
                        Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch
                    )
                    
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            print(f"Parameter values -> Delta_A: {Delta_A}, Delta_B: {Delta_B}, rho_A: {rho_A}, rho_B: {rho_B}, rho_V: {rho_V}, W: {W}")
            if not is_positive_definite(K):
                smallest_eigenvalue = torch.linalg.eigvalsh(K).min().item()
                print("Warning: K is not positive definite.  Smallest Eigenvalue ", smallest_eigenvalue)
            break
        
        loss_histories.append(total_nll)
        # Check for convergence for early stopping
        if total_nll < best_loss - tolerance:
            best_loss = total_nll
            epochs_no_improve = 0
            for name, param in zip(param_names, params):
                best_params[name] = param.clone()
            
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
            return params, best_params, loss_histories
            
    # After optimization
    if halving_log!=[]:
        print("\nLearning Rate Halving Log:", halving_log)
    
    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma)
    # K = compute_matern_covariance(alpha_matrix, nu_matrix, sigma_matrix, X)
    K = approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X_groups[-1]) # 20 Oct 2024 updated approximated matern computation
    if not is_positive_definite(K):
        smallest_eigenvalue = torch.linalg.eigvalsh(K).min().item()
        print("Warning: K is not positive definite.  Smallest Eigenvalue ", smallest_eigenvalue)    
    return params, best_params, loss_histories

def testing_for_optimize_cross_parameters_in_groups():
    # Test setup: mock or minimal input data
    p = 3  # Number of variables or features
    group_size = 5
    dimensions = 2
    device = 'cpu'  # Assume testing on CPU
    optimized_marginal_params = [torch.rand(3) for _ in range(p)]
    lr_set = {name: torch.tensor(0.1, dtype=torch.float64) for name in ['Delta_A', 'Delta_B', 'rho_A', 'rho_B', 'rho_V', 'W']}
    X_groups = [torch.rand(group_size, dimensions) for _ in range(3)]
    Y_groups = [torch.rand(group_size, p) for _ in range(3)]
    
    # Define the individual test cases as local functions
    def test_initialization():
        """Checks if the function initializes and returns expected outputs."""
        result_params, best_params = optimize_cross_parameters_in_groups(
            optimized_marginal_params, lr_set, X_groups, Y_groups, number_of_cycles=1, logging=True
        )
        
        assert isinstance(result_params, list), "Result params should be a list."
        assert isinstance(best_params, dict), "Best params should be a dictionary."
        assert all(isinstance(param, torch.Tensor) for param in result_params), "Each param should be a torch.Tensor."
        print("Initialization test passed.")

    def test_parameters_on_device():
        """Checks if the parameters are on the correct device and have gradients enabled."""
        result_params, _ = optimize_cross_parameters_in_groups(
            optimized_marginal_params, lr_set, X_groups, Y_groups, number_of_cycles=1
        )
        for param in result_params:
            assert param.device.type == device, "Parameter not on correct device."
            assert param.requires_grad, "Parameter should require gradients."
        print("Parameter device and gradient check passed.")

    def test_perform_optimization_step():
        """Tests the perform_optimization_step_with_halving function using mock inputs."""
        initial_lr = 0.1
        optimizer = torch.optim.Adam([torch.randn(1, requires_grad=True)], lr=initial_lr)
        model_state_before_step = {"param": torch.tensor(1.0, requires_grad=True)}
        grad_state_before_step = {"param_grad": torch.tensor(0.5)}

        success = perform_optimization_step_with_halving(
            optimizer=optimizer,
            params=[torch.randn(1, requires_grad=True)],
            param_names=["param"],
            model_state_before_step=model_state_before_step,
            grad_state_before_step=grad_state_before_step,
            max_halving_attempts=3,
            epoch=0,
            step=0,
            halving_log=[],
            lr_log=[],
            Delta_A=torch.tensor(1.0, requires_grad=True),
            Delta_B=torch.tensor(1.0, requires_grad=True),
            rho_A=torch.tensor(0.5, requires_grad=True),
            rho_B=torch.tensor(0.5, requires_grad=True),
            rho_V=torch.tensor(0.0, requires_grad=True),
            W = torch.full((p,), torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True),
            alpha=torch.tensor([1.0, 1.0, 1.0]),
            nu=torch.tensor([1.0, 1.0, 1.0]),
            sigma=torch.tensor([1.0, 1.0, 1.0]),
            X_batch=torch.rand(group_size, dimensions)
        )
        
        assert success is True, "Optimization step with halving failed unexpectedly."
        print("Optimization step with halving test passed.")

    def test_early_stopping():
        """Tests if early stopping occurs when there's no improvement in loss."""
        result_params, best_params = optimize_cross_parameters_in_groups(
            optimized_marginal_params, lr_set, X_groups, Y_groups,
            number_of_cycles=5,  # Small number of cycles
            steps_per_batch=1,
            print_early_stopping_epochs=True,
            logging=True
        )
        
        assert result_params, "Result params should not be empty."
        assert best_params, "Best params should not be empty."
        print("Early stopping test passed.")

    def test_time_limit():
        """Tests if the function respects the max_time_hours limit."""
        start_time = time.time()
        result_params, best_params = optimize_cross_parameters_in_groups(
            optimized_marginal_params, lr_set, X_groups, Y_groups,
            max_time_hours=0.0001,  # Small time limit
            logging=True
        )
        end_time = time.time()

        # Check if the function exited close to the specified time
        assert (end_time - start_time) < (0.0001 * 3600 + 60), "Function did not exit within the time limit."
        print("Time limit check passed.")

    def test_convergence_tolerance():
        """Tests if the function stops updating parameters once changes are below tolerance."""
        tolerance = 1e-15  # High tolerance for test purposes
        result_params, best_params = optimize_cross_parameters_in_groups(
            optimized_marginal_params, lr_set, X_groups, Y_groups,
            logging=True
        )
        
        # Check if the best params meet the convergence criteria
        assert all((abs(param - best_param) < tolerance).all for param, best_param in zip(result_params, best_params.values())), \
            "Parameters did not converge within tolerance."
        print("Convergence tolerance check passed.")

    # Run each test case
    print("Running tests for optimize_cross_parameters_in_groups...")
    test_initialization()
    test_parameters_on_device()
    # test_perform_optimization_step()
    test_early_stopping()
    test_time_limit()
    test_convergence_tolerance()
    print("All tests passed for optimize_cross_parameters_in_groups.")
# testing_for_optimize_cross_parameters_in_groups()


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
                    
                    # Step 3: Add a small noise for numerical stability; put this code inside the kernel function. # is this still needed?
                    # K += torch.eye(K.size(0), device = K.device)*1e-8
                    # K = (K + K.mT) / 2 # this is deeply problematic.  there is absolutely no reason to think that K is symmetric here.
    
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
                        if cross_psd_condition_checker(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch):
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
                smallest_eigenvalue = torch.linalg.eigvalsh(K).min().item()
                print("Warning: K is not positive definite.  Smallest Eigenvalue ", smallest_eigenvalue)

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
#                         if cross_psd_condition_checker(Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma, X_batch):
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