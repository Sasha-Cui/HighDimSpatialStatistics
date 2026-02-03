We are implementing the following contest between the codes from Joe Guinness and the codes we wrote ourselves.

The steps to this comparison follow.
1. We use the firsth half of 32_SyntheticDataCreation to generate 300 CSV files.  They are stored under ~/project/synthetic_data.
2. We use RStudio to run 30_JoeGuinnessScript.R, which loads the CSV files and returns the estimates.  These estimates are stored under ~/project/R_processed_data
3. Lastly, we go back to the second half of 32_SyntheticDataCreation to check how well the R code does on these csv files.
4. We run 33_DirectContestwR, which loads the CSV files and returns the estimates.  These estimates are stored under ~/project/python_processed_data.



By the way, I should note that 33 contains the latest code for optimisation.  We should probably confirm the parameters for the optimisation algorithm soon.   We kept changing these.  For now,

1) Individual norm_type = 'inf' Gradient Clipping for the parameters.
2) rho_A,B >0 bounds.
3) Confirm the various nugget terms.

We should not be modifying these things too much afterwards.