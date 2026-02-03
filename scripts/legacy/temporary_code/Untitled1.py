# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/Untitled1.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Create the DataFrame
data = {
    'File': [1, 2, 4, 11, 14, 17, 28, 29, 31, 32, 34, 35, 38, 41, 44],
    'Metric': [
        0.4851797137979485, 0.3304306244341856, 0.4205691512059789,
        0.5529444725072723, 0.4448945521877796, 0.44666093866479445,
        0.5629834121478279, 0.5696192010769291, 0.5535236275439431,
        0.5695402984610004, 0.5216698647360039, 0.5694269930203387,
        0.5659406736013709, 0.5661676849080745, 0.5659513959622372
    ],
    'Alpha lr': [
        0.015, 0.015, 0.015, 0.0205, 0.0205, 0.0205,
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
        0.25, 0.25, 0.25
    ],
    'Nu lr': [
        0.015, 0.015, 0.0205, 0.015, 0.0205, 0.026,
        0.025, 0.025, 0.25, 0.25, 2.5, 2.5,
        0.025, 0.25, 2.5
    ],
    'Sigma lr': [
        0.00225, 0.003, 0.00225, 0.003, 0.003, 0.003,
        0.03, 0.3, 0.03, 0.3, 0.03, 0.3,
        0.3, 0.3, 0.3
    ]
}

df = pd.DataFrame(data)

# Plot histograms
parameters = ['Alpha lr', 'Nu lr', 'Sigma lr']
for param in parameters:
    plt.figure(figsize=(8, 6))
    plt.hist(df[param], bins=10, edgecolor='black')
    plt.title(f'Histogram of {param}')
    plt.xlabel(param)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# %%
for key in ["Alpha lr", "Nu lr", "Sigma lr"]:
    print(sum(data[key])/len(data[key]))

# %%
pass

# %%
pass

