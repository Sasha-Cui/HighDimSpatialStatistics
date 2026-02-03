estimated_parameters.csv come from notebook 33 using the optimisation codes that do not contain lr halving. 
estimated_parameters_2.csv is generated 17 Sept with no clipping of gradients and with reduced stepsize for bessel function gradient.  This is meant to reduce the times that the code fails.
estimated_parameters_3.csv is applied to Hattie's multiple dataset data with old gene list (size=10)
estimated_parameters_4.csv is applied to Hattie's multiple dataset data with updated gene list (size=22)

synthetic_estimated_parameters_1.csv is on synethetic, run to incorperate the functionality of halving the lr adaptively
fitted_parameters_2 comes from 33_DirectContestwROut.  It is our finalised estimation of the trivariate Genton data set.  We are using this as a benchmarking against the Table 2 in their paper.

1 Oct I implemented a new naming convention.  epilogue.py will automatically save to fitted_parameters_i.csv.  It is however important to always keep track of which ones are which.  Therefore I should always get back to this place and write down the relevant documentations. 

fitted_parameters_0 comes from 36_SubsamplingwReplacementDayOut    It contains 25 runs of 50 locations at 22 genes.  
fitted_parameters_1 are obtained on 18 Oct 2024.  It contains 22 marginal fitting results. 