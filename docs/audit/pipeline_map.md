# Pipeline Map (Current Notebooks)

All notebooks are now stored under `notebooks/legacy/` with their original relative paths.
This map is used to define the new script/module layout.

## Stage 1: Data Generation & Preprocessing
- Synthetic data creation: `32_SyntheticDataCreation.ipynb`, `41_create_synthetic_data.ipynb`
- Real data creation: `51_create_real_data.ipynb`

## Stage 2: Kernel Smoothing & Subsampling
- Synthetic kernel smoothing: `42_kernel_smoother.ipynb`
- Synthetic subsampling: `42_random_subsampler.ipynb`
- Real-data kernel smoothing: `52_kernel_smoothing_real_data.ipynb`

## Stage 3: Model Fitting & Validation
- Synthetic fitting/validation: `43_fitting_and_validation_metric.ipynb`
- Synthetic fitting/validation (randomly subsampled): `43_randomly_subsampled_fitting_and_validation_metric.ipynb`
- Real-data fitting/validation: `53_fitting_and_validation_metric_real_data.ipynb`

## Stage 4: Cross-Term Fitting
- Cross fitting: `47_cross_fitting.ipynb`, `40_CrossFitting.ipynb`

## Stage 5: Metrics & Evaluation
- Metric calculation: `44_metric_calculation.ipynb`
- Metric calculation (randomly subsampled): `44_randomply_subsampled_metric_calculation.ipynb`

## Stage 6: Hyperparameter Analysis
- Hyperparameter visualization: `45_visualisation_of_hyperparams.ipynb`
- Metric vs hyperparameter analysis: `46_basic_data_analysis_metric_against_hyperparam.ipynb`
- Best-parameter search (real data): `54_finding_best_param_real_data.ipynb`

## Stage 7: Performance & Profiling
- Memory tracking: `37_MemoryTracking.ipynb`
- Marginal fitting benchmarks: `38_marginal_fitting.ipynb`
- CPU/GPU benchmarking: `39_cpu.ipynb`, `39_gpu.ipynb`

## Additional Top-Level Notebooks
- Bivariate/GP theory or exploration: `16_BivariateDEAtretic_Linux.ipynb`, `19_basic_gp_linux.ipynb`, `21_BivariateKernel.ipynb`
- R comparison/benchmarking: `33_DirectContestwR.ipynb`, `33_DirectContestwROut-Copy1.ipynb`
- Multi-dataset exploration: `35_HattieMultipleData.ipynb`
- Bootstrap/pseudo-distance: `61_nonparametric_boostrap.ipynb`, `62_pseudo_distance.ipynb`, `63_pseudo_distance.ipynb`

## Legacy/Experimental Areas (Converted but Not Canonical)
- `archived_code/` (100 notebooks)
- `temporary_code/` (23 notebooks)
- `ping_luo/` (22 notebooks)
- `Code Packages/` (11 notebooks, often 3rd-party or exploratory)
