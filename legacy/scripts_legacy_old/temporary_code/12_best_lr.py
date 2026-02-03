# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/12_best_lr.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
%run -i ~/project/preambles
%run -i ~/project/temporary_code/helper_functions
%run -i ~/project/temporary_code/fitting_functions

# %%
%load_ext memory_profiler
import itertools

# %%
# # Global parameters:
number_of_cycles = 1000 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
steps_per_batch = 1
dims = 2  # 2D spatial
head = 5000 # how many locations to consider in the real data set.

# %%
puck_list = ['Puck_230517_39'] # the largest puck
gene_list = [ "Inha"]
adata, X,Y, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
%%memit
start_time = time.time()
optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch, print_early_stopping_epochs = True)
end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:.6f} hours")

# %%
# ## Let us look back at the synthetic data set

# %%
# # Global parameters:
number_of_cycles = 1000 # how many passes through the training data we go through
number_of_groups = 1 # divide the data set into smaller ones, to make fitting easier.
locations_per_group = 200 # how many locations to observe per group
number_of_locations = number_of_groups * locations_per_group # total locations
number_of_simulations = 300 # for synthetic data, how many optimisation to average over
steps_per_batch = 1
dims = 2  # 2D spatial
p =  3 # how many features 

# %%
X,Y,true_K, alpha_matrix_true, nu_matrix_true, sigma_matrix_true = Genton_parametrisation(number_of_locations, dims)
ground_truth_df = store_as_df(alpha_matrix_true, nu_matrix_true, sigma_matrix_true)
estimated_params_df = pd.DataFrame()
distance_K_df = pd.DataFrame()
for _ in range(1,301):
    print(f"data set {_}")
    try:
        df = pd.read_csv(f'~/project/synthetic_data/realisation_{_}.csv')
        X = torch.tensor(df[['0','1']].values[:200], dtype=torch.float64).detach()
        expression = df['Expression'].values.reshape(200, 3)
        Y = torch.tensor(expression, dtype=torch.float64).detach()
        torch.autograd.set_detect_anomaly(True)
        optimized_marginal_params = optimize_marginal_parameters(X, Y, number_of_groups,  number_of_cycles, steps_per_batch)
        print(optimized_marginal_params)
    except Exception as e:
        print(e)
        continue

# %%
optimize_marginal_parameters(X, Y, number_of_groups=1, number_of_cycles=200, steps_per_batch=1)

# %%
ground_truth_df

