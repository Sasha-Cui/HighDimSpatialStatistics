# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/05_number_of_features.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import numpy as np
def number_of_var(p):
    return p*(p+1)/2


# %%
for p in ps:
    print(f"{number_of_var(p):.0f}")

# %%
# Load the memory profiler magic
%load_ext memory_profiler

# %%
%%memit
# Creating a large list
large_list_1 = [i for i in range(10**7)]

# Creating another large list
large_list_2 = [i * 2 for i in range(10**7)]

# %%
# Use %memit to measure memory usage for a specific line
%memit large_list_1 = [i for i in range(10**7)]

# Measure another specific operation
%memit large_list_2 = [i * 2 for i in range(10**7)]

