# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/13_parallel_processing.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
# # Testing if parallel processing really speeds things up

# %%
import os
import pandas as pd
import torch
import itertools
import multiprocessing as mp
import time

%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions
import multiprocessing as mp

# Your existing parameters
number_of_cycles, steps_per_batch = 250, 1
input_dir_base = '~/project/42_subsampled_synthetic_data'
output_dir_base = '~/project/43_estimation_results'

learning_rates = [
    {'alpha_lr': 0.00001, 'nu_lr': 0.0005, 'sigma_lr': 0.005},
    {'alpha_lr': 0.00005, 'nu_lr': 0.001, 'sigma_lr': 0.01}
]
initializations = [
    {'alpha_init': 0.001, 'nu_init': 0.5, 'sigma_init': 0.5},
    {'alpha_init': 0.005, 'nu_init': 1.0, 'sigma_init': 1.0}
]

# Define the task function
def run_optimization(l, lr_set, init_set):
    optimized_dir = os.path.expanduser(f'{output_dir_base}/{l}')
    os.makedirs(optimized_dir, exist_ok=True)

    i = 0
    while os.path.exists(os.path.expanduser(f'{input_dir_base}/{i}')):
        X_groups, Y_groups = [], []
        k = 0
        while True:
            x_path = os.path.expanduser(f'{input_dir_base}/{i}/X_subsampled_{k}.csv')
            y_path = os.path.expanduser(f'{input_dir_base}/{i}/Y_subsampled_{k}.csv')
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                break
            X_groups.append(torch.tensor(pd.read_csv(x_path).values, dtype=torch.float64))
            Y_groups.append(torch.tensor(pd.read_csv(y_path).values, dtype=torch.float64))
            k += 1

        hyperparameters_path = os.path.expanduser(f'{output_dir_base}/hyperparameters_{l}.csv')
        pd.DataFrame([lr_set | init_set]).to_csv(hyperparameters_path, index=False)

        optimized_params, best_params, loss_histories = optimize_marginal_parameters_in_groups(
            lr_set, init_set, X_groups, Y_groups, number_of_cycles, steps_per_batch
        )
        
        for j, (optimized_params_j, best_params_j) in enumerate(zip(optimized_params, best_params)):
            optimized_params_j = [v.item() if isinstance(v, torch.Tensor) else v for v in optimized_params_j]
            best_params_j = [v.item() if isinstance(v, torch.Tensor) else v for v in best_params_j[0].values()]

            optimized_params_path = f'{optimized_dir}/optimized_parameters_dataset_{i}_feature_{j}.csv'
            os.makedirs(os.path.dirname(optimized_params_path), exist_ok=True)
            # pd.DataFrame([optimized_params_j], columns=['alpha', 'nu', 'sigma']).to_csv(optimized_params_path, index=False)

            best_params_path = f'{optimized_dir}/best_parameters_dataset_{i}_feature_{j}.csv'
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            # pd.DataFrame([best_params_j], columns=['alpha', 'nu', 'sigma']).to_csv(best_params_path, index=False)

            loss_histories_path = f'{optimized_dir}/loss_histories_dataset_{i}_feature_{j}.csv'
            os.makedirs(os.path.dirname(loss_histories_path), exist_ok=True)
            # if j < len(loss_histories):
                # pd.DataFrame({'loss': list(loss_histories[j])}).to_csv(loss_histories_path, index=False)
        i += 1
        if i == 2:
            break

# Test function for serial and parallel processing
def test_serial():
    for l, (lr_set, init_set) in enumerate(itertools.product(learning_rates, initializations)):
        run_optimization(l, lr_set, init_set)

def test_parallel():
    combinations = [(l, lr_set, init_set) for l, (lr_set, init_set) in enumerate(itertools.product(learning_rates, initializations))]
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        pool.starmap(run_optimization, combinations)

if __name__ == '__main__':
    # Measure serial processing time
    start_time = time.time()
    test_serial()
    serial_time = time.time() - start_time
    print(f"Serial processing time: {serial_time:.2f} seconds")

    # Measure parallel processing time
    start_time = time.time()
    test_parallel()
    parallel_time = time.time() - start_time
    print(f"Parallel processing time: {parallel_time:.2f} seconds")

    # Report speedup
    speedup = serial_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

