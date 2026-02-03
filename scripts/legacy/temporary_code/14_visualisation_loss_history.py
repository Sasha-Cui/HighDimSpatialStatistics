# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/14_visualisation_loss_history.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the loss history data
l=0
# i=0
# j=0
for i in range(2):
    for j in range(22):
        
        file_path = os.path.expanduser(f"~/project/53_hattie_marginal_estimation_results/{l}/loss_histories_dataset_{i}_feature_{j}.csv")
        loss_history = pd.read_csv(file_path)
        
        # Plot the loss history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label="Loss", marker="o")
        plt.title(f"Loss History for Hyperparam {l}, Dataset {i}, Feature {j}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss Value")
        # plt.xscale("log")
        plt.legend()
        plt.grid()
        plt.tight_layout()

# %%
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Load the loss history data
# l=5
# # i=0
# # j=0
# for i in range(2):
#     for j in range(3):
#         file_path = os.path.expanduser(f"~/project/44_estimation_results/{l}/loss_histories_dataset_{i}_feature_{j}.csv")
#         loss_history = pd.read_csv(file_path)
        
#         # Plot the loss history
#         plt.figure(figsize=(10, 6))
#         plt.plot(loss_history, label="Loss", marker="o")
#         plt.title(f"Loss History for Hyperparam {l}, Dataset {i}, Feature {j}")
#         plt.xlabel("Iteration")
#         plt.ylabel("Loss Value")
#         # plt.xscale("log")
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

successful_hyperparameters = []
# Load the loss history data
for l in range(50):
    for i in range(2):
        for j in range(3):
            try:
                file_path = os.path.expanduser(f"~/project/archived_code/43_estimation_results/{l}/loss_histories_dataset_{i}_feature_{j}.csv")
                loss_history = pd.read_csv(file_path)
                if loss_history["loss"].iloc[-1]<4558:
                    successful_hyperparameters.append(l)
                    print (pd.read_csv(f"~/project/archived_code/43_estimation_results/validation_metric_hyperparam_{l}.csv")["average_validation_metric"][0])
            except Exception as e:
                # print(f"an error occured at {l,i,j}")
                continue

# %%
list(set(successful_hyperparameters))

# %%
loss_history["loss"].iloc[-1]

