Hi. 

This file gives a brief idea of what the different files do.  While reading this note, you should list the files alphabetically.

FOLDERS
With a few exceptions, almost all folders contain different types of data.  The project involves real data, processed real data, synthetic data, processed synthetic data.  The source data coming from Professor Hattie Chung's 2024 preprint are in 

    mouse_ovary_slide_seq_young_estrus.h5ad
    ovary_Puck_230517_39.h5ad

.  The other data are all under folders starting 

    41_1_train_data
    ...
    53_hattie_marginal_estimation_results

A large number of codes are under 
    archived_code
.  Unfortunately, they have been ruled out as less relevant for various reasons. 

Various notebooks for short-term purposes are stored under
    temporary_code
.  They are quite useful, but they are not central to the project pipeline. 

The folder
    ping_luo
contains code that I sent to Ms Ping Luo for her to check why the codes are slower on GPU and faster on CPU.  She said that it's normal. 

The foler
    GpGp_multi_paper
contains an R package from Joe Genuiness.  While this code runs slowly, Joe is the only person who agreed to give me any code at all.  All other authors that I contacted have told me that they lost the code, so their empirical results are highly questionable.  If you cannot make your method perform as well as they claim in their papers, then that's expected.  To this this R-package, you need the folder 
    R
and the results are in 
    R_processed_data
.  You can use this for benchmarking. 

There are various packages that need to be installed.  The most useful is pytorch_env.yml, for this allows you to run the code on GPU.  No guarantee that it's faster there, though.  They are under the folder
    Environments 

The folders 
    Papers
    Books
contain many relevant pieces of literature.  You should probably start with asking ChatGPT to summarise what they are about before reading them. 


The folder
    Code Packages
contains many existing packages.  They are usually highly painful to install or use.  Sadly, I've not made them to run.

The files that end with .sh are for running the corresponding notebooks in the cluster.



JUPYTER NOTEBOOKS STRUCTURE
In general, the older the notebook is, the more outdated they are.  The old notebooks are rejected for all kinds of reasons.  

1. Data Generation and Preprocessing

This is the first stage where data is either generated synthetically or preprocessed from real datasets.
Synthetic Data:

    32_SyntheticDataCreation.ipynb → Generates synthetic spatial gene expression data.
    41_create_synthetic_data.ipynb → Another version focused on synthetic dataset creation.

Real Data Preparation:

    51_create_real_data.ipynb → Loads real-world spatial gene expression datasets from pucks and organizes them for analysis.

Kernel Smoothing and Subsampling (Both Real and Synthetic Data):

    42_kernel_smoother.ipynb → Applies kernel smoothing to synthetic data.
    42_random_subsampler.ipynb → Randomly subsamples synthetic data.
    52_kernel_smoothing_real_data(1).ipynb → Applies kernel smoothing to real datasets.

2. Model Fitting and Validation

Once the data is ready, the next set of notebooks focuses on optimizing model parameters and validating performance.
Fitting Models (Synthetic Data):

    43_fitting_and_validation_metric.ipynb → Performs model fitting and computes validation metrics.
    43_randomly_subsampled_fitting_and_validation_metric.ipynb → Repeats the process for randomly subsampled datasets.

Fitting Models (Real Data):

    53_fitting_and_validation_metric_real_data.ipynb → Runs model fitting and validation metrics on real gene expression datasets.

Cross-Term Fitting:

    47_cross_fitting.ipynb → Takes the optimized marginal parameters and performs cross-term fitting (adjusting interdependencies between genes).

3. Evaluating Model Performance and Hyperparameter Tuning

After fitting the models, the next logical step is to evaluate how well the models performed and fine-tune hyperparameters.
Metric Calculation:

    44_metric_calculation.ipynb → Computes evaluation metrics for the fitted synthetic models.
    44_randomply_subsampled_metric_calculation.ipynb → Similar, but for randomly subsampled data.

Hyperparameter Analysis:

    45_visualisation_of_hyperparams.ipynb → Visualizes hyperparameter effects on model performance.
    46_basic_data_analysis_metric_against_hyperparam.ipynb → Analyzes how different hyperparameter settings impact the model.

Finding the Best Hyperparameters:

    54_finding_best_param_real_data.ipynb → Systematically searches for the best hyperparameters for real data.
    53_fitting_and_validation_metric_real_data.ipynb → Stores estimated hyperparameters and their associated validation metrics.

4. Computational Performance Optimization

A unique and important aspect of the work is the focus on memory usage and computation time tracking to optimize large-scale computations.
Tracking Memory and Benchmarking Computation:

    37_MemoryTracking.ipynb → Tracks memory usage during optimization.
    38_marginal_fitting.ipynb → Fits models for subsets of genes while tracking computational performance.
    39_cpu.ipynb → Runs the model on a CPU for benchmarking.
    39_gpu.ipynb → Runs the model on a GPU to compare performance.

This is useful because the computing cluster allowance is extremely low and you have to be very careful or else they will end up failing to complete the execution of the codes.

5. Core Python Functions

Three Python script files provide essential functions used throughout multiple notebooks:

    preambles.py → Contains all required imports and initial configurations.
    helper_functions.py → Provides utility functions used across different notebooks.
    fitting_functions.py → Implements the core fitting functions for parameter estimation.

These scripts ensure code reusability and prevent unnecessary duplication in different notebooks.  These three files form the core of the useful work in my opinion. 



INDIVIDUAL NOTEBOOKS
Synthetic Data Generation

    32_SyntheticDataCreation.ipynb
        Purpose: Generates synthetic spatial gene expression data for benchmarking models.
        Key Steps:
            Defines a structured spatial grid.
            Simulates gene expression values using a predefined probabilistic model.
            Saves synthetic datasets for later model training and testing.

    41_create_synthetic_data.ipynb
        Purpose: Generates synthetic spatial gene expression datasets.
        Key Steps:
            Creates a dataset with spatial gene expression patterns.
            Saves the generated data in .csv format for further processing.

Data Processing

    42_kernel_smoother.ipynb
        Purpose: Applies kernel smoothing to preprocess synthetic data.
        Key Steps:
            Uses a Gaussian kernel to smooth spatial gene expression data.
            Stores smoothed data for downstream analysis.

    42_random_subsampler.ipynb
        Purpose: Randomly subsamples the dataset to create different training sets.
        Key Steps:
            Selects a subset of the data based on predefined criteria.
            Ensures a balanced distribution of spatial points.

Model Fitting and Validation (Synthetic Data)

    43_fitting_and_validation_metric.ipynb
        Purpose: Fits models and calculates validation metrics.
        Key Steps:
            Optimizes model parameters.
            Computes performance metrics and loss functions.

    43_randomly_subsampled_fitting_and_validation_metric.ipynb
        Purpose: Same as above, but applied to randomly subsampled datasets.

    44_metric_calculation.ipynb
        Purpose: Computes evaluation metrics for model predictions.

    44_randomply_subsampled_metric_calculation.ipynb
        Purpose: Computes validation metrics on subsampled datasets.

Hyperparameter Analysis

    45_visualisation_of_hyperparams.ipynb
        Purpose: Visualizes the effect of hyperparameters on model performance.

    46_basic_data_analysis_metric_against_hyperparam.ipynb

    Purpose: Analyzes how different hyperparameters affect model performance using regression and correlation analysis.

    47_cross_fitting.ipynb

    Purpose: Fits cross-term parameters after marginal optimization.
    Key Steps:
        Runs optimization over different hyperparameter settings.
        Stores both Genton parameters (ρ, Δ_A, Δ_B, W) and Matérn parameters (α, ν, σ).
        Computes covariance matrices and validation metrics.

Computational Performance Tracking

    37_MemoryTracking.ipynb

    Purpose: Tracks memory usage during optimization.
    Key Steps:
        Uses tracemalloc to monitor memory consumption.
        Runs an optimization step with limited iterations to measure resource usage.

    38_marginal_fitting.ipynb

    Purpose: Fits model parameters for subsets of genes separately.

    39_cpu.ipynb

    Purpose: Runs model fitting on CPU to benchmark performance.

    39_gpu.ipynb

    Purpose: Runs model fitting on GPU to compare with CPU performance.

Real Data Processing

    51_create_real_data.ipynb

    Purpose: Prepares real gene expression datasets.
    Key Steps:
        Loads spatial gene expression data from pucks.
        Saves preprocessed data for further analysis.

    52_kernel_smoothing_real_data(1).ipynb

    Purpose: Applies kernel smoothing to real gene expression data.

    53_fitting_and_validation_metric_real_data.ipynb

    Purpose: Fits models on real data and computes validation metrics.
    Key Steps:
        Reads kernel-smoothed data.
        Fits models and estimates parameters (α, ν, σ).
        Stores fitted parameters and loss histories.

    54_finding_best_param_real_data.ipynb

    Purpose: Finds the best hyperparameter setting based on loss minimization.
    Key Steps:
        Iterates through hyperparameters.
        Selects the best parameter setting for each feature.

Python Script Summaries

    fitting_functions.py

    Purpose: Implements core fitting routines for model optimization.
    Key Functions:
        optimize_marginal_parameters() – Optimizes parameters for individual genes.
        optimize_cross_parameters() – Optimizes cross-correlation parameters.

    helper_functions.py

    Purpose: Provides utility functions for data processing and computation.

    preambles.py

    Purpose: Contains necessary imports, paths, and configurations.

Finally, I have created a git version as of 3 Feb 2025, so you can always restore it to this version.  Good luck!

Regards,
Sasha Cui 
3 Feb 2025