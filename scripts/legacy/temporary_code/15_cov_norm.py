# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/15_cov_norm.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import os
import numpy as np
import pandas as pd

def calculate_frobenius_norm(i, j):
    """
    Reads a CSV file and calculates its Frobenius norm.
    
    Parameters:
        i (int): First identifier in the file path.
        j (int): Second identifier in the file path.
        
    Returns:
        float: Frobenius norm of the matrix.
    """
    # Construct the file path
    file_path = os.path.expanduser(f"~/project/41_3_test_cov/K_test_{i}_{j}.csv")
    
    try:
        # Read the CSV file
        matrix = pd.read_csv(file_path).values
        # Calculate the Frobenius norm
        frobenius_norm = np.linalg.norm(matrix, 'fro')
        print(f"Frobenius norm for K_test_{i}_{j}.csv: {frobenius_norm}")
        return frobenius_norm
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
# Replace i and j with specific values
for i in range(2):
    for j in range(3):
        calculate_frobenius_norm(i, j)

# %%
pass

