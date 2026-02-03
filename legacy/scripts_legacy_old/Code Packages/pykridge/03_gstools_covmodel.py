# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/Code Packages/pykridge/03_gstools_covmodel.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
#
# # GSTools Interface
#
# Example how to use the PyKrige routines with a GSTools CovModel.

# %%
import gstools as gs
import numpy as np
from matplotlib import pyplot as plt

from pykrige.ok import OrdinaryKriging

# conditioning data
data = np.array(
    [
        [0.3, 1.2, 0.47],
        [1.9, 0.6, 0.56],
        [1.1, 3.2, 0.74],
        [3.3, 4.4, 1.47],
        [4.7, 3.8, 1.74],
    ]
)
# grid definition for output field
gridx = np.arange(0.0, 5.5, 0.1)
gridy = np.arange(0.0, 6.5, 0.1)
# a GSTools based covariance model
cov_model = gs.Gaussian(dim=2, len_scale=4, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
# cov_model = gs.Matern(dim=2, var=1, len_scale=10, nu=1.5)
# ordinary kriging with pykrige
OK1 = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], cov_model)
z1, ss1 = OK1.execute("grid", gridx, gridy)
plt.imshow(z1, origin="lower")
plt.show()


# Some of the GSTools-based covariance models include:

#     Exponential Model:
#         The exponential model is commonly used to represent spatial correlation. It has a rapidly decreasing correlation with distance.

#     python

# import gstools as gs

# model = gs.Exponential(dim=2, var=1, len_scale=10)

# Gaussian Model:

#     The Gaussian model is another popular choice for spatial correlation. It has a bell-shaped correlation function.

# python

# import gstools as gs

# model = gs.Gaussian(dim=2, var=1, len_scale=10)

# Matérn Model:

#     The Matérn model is a generalization of the Gaussian model and includes a smoothness parameter (nu).

# python

# import gstools as gs

# model = gs.Matern(dim=2, var=1, len_scale=10, nu=1.5)

# Spherical Model:

#     The spherical model is characterized by a correlation that quickly reaches a constant value beyond a certain range.

# python

# import gstools as gs

# model = gs.Spherical(dim=2, var=1, len_scale=10)

# Cubic Model:

#     The cubic model is less smooth compared to the Gaussian model and has a more abrupt cutoff.

# python

# import gstools as gs

# model = gs.Cubic(dim=2, var=1, len_scale=10)

# Linear Model:

#     The linear model represents a linearly decreasing correlation with distance.

# python

#     import gstools as gs

#     model = gs.Linear(dim=2, var=1, len_scale=10)

# These examples demonstrate how to create covariance models using GSTools. Adjust the parameters such as dimension (dim), variance (var), and length scale (len_scale) based on the characteristics of your spatial data.

# Remember to consult the GSTools documentation for more details and additional covariance models: GSTools Documentation.

# %%
pass

