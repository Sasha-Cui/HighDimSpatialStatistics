## the file saver
# Generate a timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Get the Slurm job ID from the environment (if running within Slurm)
slurm_job_id = os.getenv('SLURM_JOB_ID', 'no_slurm_id')

# Construct a more descriptive filename
filepath = os.path.expanduser(f"~/project/python_processed_data/{notebook_name}_job_{slurm_job_id}_time_{timestamp}.csv")

# Save the DataFrame to the new CSV file
df_to_plot.to_csv(filepath, index=False)
print(f"DataFrame saved to {filepath}")


# 22 Oct 2024 updated to be more descriptive
# filename_counter = 0
# # Loop to find the first available file name
# while True:
#     filepath = os.path.expanduser(f"~/project/python_processed_data/fitted_parameters_{filename_counter}.csv")
#     if not os.path.exists(filepath):
#         # File does not exist, we can use this file name
#         break
#     filename_counter += 1

# # Save the DataFrame to the new CSV file
# df_to_plot.to_csv(filepath, index=False)
# print(f"DataFrame saved to {filepath}")



if histograms_are_plotted == True:
    columns = df_to_plot.columns
    # Calculate the number of rows and columns for subplots
    n_params = len(columns)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)  # Calculate required rows based on the number of parameters
    
    plt.figure(figsize=(15, 3 * n_rows))  # Adjust height based on number of rows
    for i, col in enumerate(columns):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot the histogram of estimates
        plt.hist(df_to_plot[col], bins=30, color='skyblue', edgecolor='black')
        
        # Plot the vertical line for the true value
        if 'ground_truth_df' in globals():
            if isinstance(ground_truth_df, pd.DataFrame):
                plt.axvline(x=ground_truth_df[col].iloc[0], color='red', linestyle='--', linewidth=2)
        
        plt.title(f'{col}')
        plt.xlabel(f'{col}')
        plt.ylabel('Frequency')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
