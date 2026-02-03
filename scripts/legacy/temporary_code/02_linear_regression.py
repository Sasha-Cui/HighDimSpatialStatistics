# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/notebooks/legacy/temporary_code/02_linear_regression.ipynb on 2026-02-03 10:31:16
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import numpy as np
import matplotlib.pyplot as plt

# Data points for memory (unit: GB)
X = np.array([100, 200, 500, 1000, 2000, 4000, 5000])
Y = np.array([1.51, 1.88, 4.54, 14.299569563940167, 52.88052634708583, 207.37584788911045, 323.18])
# Perform a polynomial fit of degree 2 (quadratic) or degree 3 for more flexibility
degree = 2  # You can change this to 3, 4, etc. for higher degree polynomials
coefficients = np.polyfit(X, Y, degree)

# Generate the polynomial function based on the coefficients
polynomial = np.poly1d(coefficients)

# Generate Y predictions from the polynomial model
X_fit = np.linspace(0, 8000, 500)  # More points to make the curve smooth
Y_pred = polynomial(X_fit)

# Print the polynomial coefficients
print(f"Polynomial coefficients (highest degree first): {coefficients}")

# Plot the original data points and the polynomial fit
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X_fit, Y_pred, color='red', label=f'Polynomial fit (degree {degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Polynomial Regression for Memory Use in GB (Degree {degree})')
plt.legend()
plt.show()

# %%
# # Data points for time (unit: Hr)
X = np.array([50,60,70,80,90,100,1000,2000,4000])
Y = np.array([0.004596,0.005010,0.005451,0.006041,0.006664,0.007388,0.372328,1.491346,6.061128])

# Perform a polynomial fit of degree 2 (quadratic) or degree 3 for more flexibility
degree = 2  # You can change this to 3, 4, etc. for higher degree polynomials
coefficients = np.polyfit(X, Y, degree)

# Generate the polynomial function based on the coefficients
polynomial = np.poly1d(coefficients)

# Generate Y predictions from the polynomial model
X_fit = np.linspace(0, 8000, 500)  # More points to make the curve smooth
Y_pred = polynomial(X_fit)

# Print the polynomial coefficients
print(f"Polynomial coefficients (highest degree first): {coefficients}")

# Plot the original data points and the polynomial fit
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X_fit, Y_pred, color='red', label=f'Polynomial fit (degree {degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Polynomial Regression for Time Use in Hrs (Degree {degree})')
plt.legend()
plt.show()

# %%
#Let us look at the case with 2000*20/p locations over p genes
X=np.array([1,2,4,5,10])
Y=np.array([0.768505, 0.879620,0.946825,.958375,.984581])
# def number_of_var(p):
#     return p*(p+1)/2

# X = number_of_var(X)

# %%
# Perform a polynomial fit of degree 2 (quadratic) or degree 3 for more flexibility
degree = 2  # You can change this to 3, 4, etc. for higher degree polynomials
coefficients = np.polyfit(X, Y, degree)

# Generate the polynomial function based on the coefficients
polynomial = np.poly1d(coefficients)

# Generate Y predictions from the polynomial model
X_fit = np.linspace(0, 20, 500)  # More points to make the curve smooth
Y_pred = polynomial(X_fit)

# Print the polynomial coefficients
print(f"Polynomial coefficients (highest degree first): {coefficients}")

# Plot the original data points and the polynomial fit
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X_fit, Y_pred, color='red', label=f'Polynomial fit (degree {degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Polynomial Regression for Time Use in Hrs (Degree {degree})')
plt.legend()
plt.show()

# %%
pass

# %%
pass

