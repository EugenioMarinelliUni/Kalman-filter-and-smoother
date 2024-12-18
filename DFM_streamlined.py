# %%
## Loading the dataset ##
import pandas as pd

file_path = 'standardized_compact_dataset.csv'
data = pd.read_csv(file_path, index_col=0, parse_dates=True)

print(data.head())
print(data)
print(type(data))
all_data_numpy_array = data.to_numpy()

print(type(all_data_numpy_array))  # Check the type

print(all_data_numpy_array.shape)

print(data)

# Step 1: Get the final date from the dataset
final_date = data.index[-1]  # Get the last date in the dataframe

# Step 2: Create a list of all monthly dates between 1-1-2000 and the final date
start_date = pd.to_datetime('2000-01-01')  # Start date
# Use 'MS' for month start frequency
date_range = pd.date_range(start=start_date, end=final_date, freq='MS')  # 'MS' for month start

# Step 3: Return the number of elements in the date_range
print(len(date_range))  # Number of monthly dates in the range

# Optionally, print the actual dates if needed
print(date_range)

# %%

### Testing code

import numpy as np
from scipy.linalg import eig

# Filter the DataFrame for dates up to December 1999 to obtain the vintages
filtered_data = data.loc[:'1999-12-31']

# Extract the values as a compact dataset
x = filtered_data.values

T, N = x.shape
print(x.shape) # 449x118 matrix
print(x)

# Output the values of T, N, and the first rows of x
print("T (Number of rows):", T)
print("N (Number of columns):", N)
print("First 5 rows and 5 columns of x:\n", x[:5, :5])

r = 2
q = 2
p = 1
nlag = p - 1

##########################

cov_x = np.cov(x, rowvar=False) # pairwise covariance between the variables
print("Size (number of elements):", cov_x.size)
print("Shape (dimensions):", cov_x.shape)
print("Number of dimensions:", cov_x.ndim)

eigvals, eigvecs = eig(cov_x)
# Print eigenvalues
print("Eigenvalues:")
print(eigvals)
print("Size (number of elements):", eigvals.size)
print("Shape (dimensions):", eigvals.shape)
print("Number of dimensions:", eigvals.ndim)

# Print eigenvectors
print("\nEigenvectors:")
print(eigvecs)
print("Size (number of elements):", eigvecs.shape)

idx = eigvals.argsort()[::-1]
print(idx)

eigvals, eigvecs = eigvals[idx][:r], eigvecs[:, idx][:, :r]
print(eigvals.shape)
print(eigvecs.shape)

F = x @ eigvecs # linear transformation of the original data into a new space defined by the eigenvectors
print(F.size)
print(F.shape) # a 449 x 2 matrix
print(F.shape)
print(F)


#########################

## Factor loadings ##

C = eigvecs[:, :r]  # Take the first r columns of eigvecs to get the factor loadings for the top r factors

print(C) # factor loadings matrix
print(C.shape)
# C_transposed = C.T
# print(C_transposed)
# print(C_transposed.shape)

## Formula: x = F * C transposed + e
#########################
print(x.shape)
residuals = x - F @ eigvecs.T
print(residuals.shape)
R = np.diag(np.diag(np.cov(residuals, rowvar=False)))

# computes the covariance matrix of the residuals
# (the difference between the original data and the transformation using eigenvectors),
# then extracts its diagonal elements, then builds a matrix having on its diagonal the elements extracted and zeroes elsewhere.
# R: covariance matrix of the idiosyncratic (residual) errors that are specific to the observed data

print(R)
print(R.shape) # 118 x 118 matrix

# %%

### Estimation of the matrix of autoregressive coefficients ###

# Example data: F is the matrix of factors with shape (T, r)
T, r = F.shape  # T: number of time intervals, r: number of factors

# Step 1: Prepare lagged matrices
Z = F[:-1, :]  # Factors excluding the last observation (shape: (T-1, r))
z = F[1:, :]   # Factors excluding the first observation (shape: (T-1, r))

# Step 2: Estimate VAR(1) coefficient matrix A
A = np.linalg.inv(Z.T @ Z) @ Z.T @ z
print("Estimated VAR(1) coefficient matrix A:")
print(A)

# Step 3: Estimate residuals and covariance matrix H
residuals = z - Z @ A
H = np.cov(residuals, rowvar=False)
print("Residuals covariance matrix H:")
print(H)

# Step 4: Validate the model (check eigenvalues of A)
eigvals = np.linalg.eigvals(A)
print("Eigenvalues of A:")
print(eigvals)

if np.all(np.abs(eigvals) < 1):
    print("The VAR(1) process is stable.")
else:
    print("The VAR(1) process is unstable; consider alternative models.")




# %%

# initx = F[0, :] # replaced with new code below

### Initialization of the latent factors using the last estimated factors
initx = F[-1, :]
print(initx.shape)
print(initx)

## Initialization of the Steady-State Covariance Matrix
## The steady-state covariance matrix (VV) represents the long-run variance of the state vector under the VAR(1) process
## For a VAR(1) process: Ft=AFt−1+ϵt,ϵt∼N(0,Q)
## The steady-state covariance matrix VV satisfies the Lyapunov equation: V=AVAT+Q


from scipy.linalg import solve_discrete_lyapunov

# Solve the Lyapunov equation for the steady-state covariance matrix
V = solve_discrete_lyapunov(A, H)

print("Steady-State Covariance Matrix V:")
print(V)

# %%
## Saving the data for Latent_factors_vintage_f_and_s

import pickle

# Check and print the data types of the variables
print("Data type of A:", type(A))
print("Data type of C:", type(C))
print("Data type of R:", type(R))
print("Data type of R:", type(V))
print("Data type of F:", type(F))
print("Data type of filtered_data:", type(filtered_data))

# Save the numpy arrays and DataFrame to a pickle file
with open('shared_variables_vintage.pkl', 'wb') as f:
    pickle.dump({'A': A, 'C': C, 'R': R, 'V': V, 'F': F, 'filtered_data': filtered_data}, f)

print("Variables and DataFrame saved to shared_variables_vintage.pkl")
# %%

import numpy as np

start_date = '2000-01-01'
data_after_vintage = data.loc[start_date:]
obs_data = data_after_vintage.to_numpy()


#########################

## Saving the output of interest in a shared file to ensure its reusability in the Kalman_filter_and_smoother file
import pickle

# Check and print the data types of the variables
print("Data type of A:", type(A))
print("Data type of C:", type(C))
print("Data type of R:", type(R))
print("Data type of all_data_numpy_array:", type(all_data_numpy_array))

# Save the numpy arrays and DataFrame to a pickle file
with open('shared_variables.pkl', 'wb') as f:
    pickle.dump({'A': A, 'C': C, 'R': R, 'all_data_numpy_array': all_data_numpy_array}, f)

print("Variables and DataFrame saved to shared_variables.pkl")

##
print(F.shape)