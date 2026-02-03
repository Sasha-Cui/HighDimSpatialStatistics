# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/04_some_visualisation.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import math, os
import pandas as pd
import matplotlib.pyplot as plt
histograms_are_plotted = True

file_path = os.path.expanduser(f'~/project/python_processed_data/fitted_parameters_0.csv')
df_to_plot = pd.read_csv(file_path)

# %%
df_to_plot

# %%
df_to_plot.std().mean()

# %%
df_to_plot.std()

# %%
import time
import pandas as pd
import matplotlib.pyplot as plt

def sleeper_histograms(df, batch_size=9, pause_time=0):
    # Iterate over the DataFrame columns in batches of size `batch_size`
    for i in range(0, df.shape[1], batch_size):
        # Select a batch of columns
        batch = df.iloc[:, i:i+batch_size]
        
        # Create a new figure for the histograms
        fig = plt.figure(figsize=(10, 8))
        
        # Plot histograms for the current batch of columns
        batch.hist(ax=fig.gca(), bins=20)
        plt.suptitle(f"Histograms for columns {i+1} to {min(i+batch_size, df.shape[1])}")
        
        # Display the histograms
        plt.show(block=False)
        plt.pause(0)
        plt.close()
        
sleeper_histograms(df_to_plot)

# %%
# import numpy as np
# import matplotlib.pyplot as plt

# def show_image(n):
#     fig, ax = plt.subplots()
#     x = np.linspace(0,1,100)
#     y = x**n
#     ax.plot(x,y, label = 'x**{}'.format(n))
#     ax.legend()
#     plt.show(block=False)
#     plt.pause(1)
#     plt.close(fig)



# for i in range(10):
#     show_image(i)

# %%
pass

