#!/usr/bin/env python
# coding: utf-8

# In[ ]:

## Preliminary steps to save the dataset from google drive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd

# Step 1: Authenticate the user
gauth = GoogleAuth()

gauth.LoadClientConfigFile('client_secret.json')  # Replace with your actual filename

gauth.LocalWebserverAuth()  # This will open a browser for authentication

# Step 2: Create a PyDrive GoogleDrive instance
drive = GoogleDrive(gauth)

# List files in your Google Drive
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()

for file in file_list:
    print(f"Title: {file['title']}, ID: {file['id']}")

# Locate folder of interest and list the files in it
folder_id = '1FMVnbFMURY8GP_LOn9yXuALR5o2EYOTZ'
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

for file in file_list:
    print(f"Title: {file['title']}, ID: {file['id']}")

import os

# Locate file in google drive directory and save it in project working directory

# Locate the file in Google Drive
file_id = '1OYspesLaTYuk3rjNm22Pd2RadsGVlTK1'  # Replace with the actual file ID
file = drive.CreateFile({'id': file_id})

# Define the path to save the file in the current working directory
working_directory = os.getcwd()
file_name = 'standardized_compact_dataset.csv'  # Name of the file to save locally
file_path = os.path.join(working_directory, file_name)

# Download and save the file
file.GetContentFile(file_path)
print(f"File downloaded and saved at: {file_path}")

###### preliminary step end #####

## Loading the dataset ##
import pandas as pd

file_path = 'standardized_compact_dataset.csv'
data = pd.read_csv(file_path, index_col=0, parse_dates=True)

print(data.head())
print(data)


# In[ ]:


### Testing code

import numpy as np
from scipy.linalg import eig, pinv, block_diag
from numpy.linalg import inv

# Filter the DataFrame for dates up to December 1999 to obtain the vintages
filtered_data = data.loc[:'1999-12-31']

# Extract the values as a compact dataset
x = filtered_data.values

T, N = x.shape
print(x.shape) # 449x118 matrix
print(x)

test = x[0, :]
print(test)

# Output the values of T, N, and the first rows of x
print("T (Number of rows):", T)
print("N (Number of columns):", N)
print("First 5 rows and 5 columns of x:\n", x[:5, :5])

r = 2
q = 2
p = 1
nlag = p - 1

##########################

cov_x = np.cov(x, rowvar=False) # covariance matrix of the vintage
print("Size (number of elements):", cov_x.size)
print("Shape (dimensions):", cov_x.shape)
print("Number of dimensions:", cov_x.ndim)
print(cov_x[:3])
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
C_transposed = C.T
print(C_transposed)
print(C_transposed.shape)

## Formula: x = F * C transposed + e
#########################

R = np.diag(np.diag(np.cov(x - F @ eigvecs.T, rowvar=False)))
# computes the covariance matrix of the residuals
# (the difference between the original data and the transformation using eigenvectors,
# then extracts its diagonal elements, then builds a matrix having on its diagonal the elements extracted and zeroes elsewhere.
# R: covariance matrix of the idiosyncratic (residual) errors that are specific to the observed data

print(R)
print(R.shape) # 118 x 118 matrix

####################################

# VAR model estimation

####################################

A_temp = np.zeros((r, r * p)) # Creates a temporary matrix filled with zeros of dimension r x r*p
I = np.eye(r * p) # Creates an indentiy matrix of size r*p x r*p

print(type(A_temp))
print("Size (number of elements):", A_temp.size)
print("Shape (dimensions):", A_temp.shape)
print("Number of dimensions:", A_temp.ndim)
print(A_temp[:2])

print(I.shape)
LL = I.shape[0]
if p != 1:
    A = np.vstack((A_temp.T, I[:LL-r, :]))  # Equivalent to rbind(A_temp, I[1:(LL-r), ])
else:
    A = np.vstack((A_temp.T, np.empty((0, r * p))))  # Equivalent to rbind(t(A_temp), I[0, ]) # Modified due to different indexing between R and Python


print("Size (number of elements):", A.size)
print("Shape (dimensions):", A.shape)
print("Number of dimensions:", A.ndim)
print(A[:3]) # 2 x 2 mtrix of zeros

Q = np.zeros((r * p, r * p)) # 2 x 2 identiy matrix. Q: covariance matrix of process noise
Q[:r, :r] = np.eye(r)
print("Size (number of elements):", Q.size)
print("Shape (dimensions):", Q.shape)
print("Number of dimensions:", Q.ndim)
print(Q[:3])

####################################

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

####################################
### Alternative method to estimate the matrix of autoregressive coefficients ###

Z = F[:-1, :] # slicing, all rows except the last row
z = F[1:, :] # slicing, start from the second row and include all subsequent rows
print(Z.shape)
print(z.shape)
print(Z)
print(z)


A_temp = inv(Z.T @ Z) @ Z.T @ z # matrix of Ordinary Least Squares (OLS) estimators used to explain z based on Z
# A[:r, :r * p] = A_temp.T # assigning a portion of A_temp.T to the first part of A ### investigate this line of code ###
print(A_temp)
print(A)

###############

e = z - Z @ A_temp  # VAR residuals
print(e.shape)
print(e)
H = np.cov(e, rowvar=False) # covariance matrix of the residuals
print(H.shape)
print(H)

Q[:r, :r] = H # covariance matrix of the process noise in the state-space model (no difference from H when r = 2)
print(Q.shape)
print(Q)
##### Alternative method ends here ###

#### Restart running code from here ####

print(A.shape)

# initx = F[0, :] # replaced with new code below

### Initialization of the latent factors using the last estimated factors
initx = F[-1, :]
print(initx.shape)
print(initx)

## Initialization of the Steady-State Covariance Matrix
## The steady-state covariance matrix (VV) represents the long-run variance of the state vector under the VAR(1) process
## For a VAR(1) process: Ft=AFt−1+ϵt,ϵt∼N(0,Q) Ft=AFt−1+ϵt,ϵt∼N(0,Q)
## The steady-state covariance matrix VV satisfies the Lyapunov equation: V=AVAT+Q


from scipy.linalg import solve_discrete_lyapunov

# Solve the Lyapunov equation for the steady-state covariance matrix
V = solve_discrete_lyapunov(A, H)

print("Steady-State Covariance Matrix V:")
print(V)

####### The Lyapunov equation "behind the scenes" (do not run) ###

kron_A = np.kron(A, A)
print(kron_A.shape)
print(kron_A)

Q_flatten = Q.flatten(order='F').reshape(-1, 1)
print(Q_flatten.shape)

diag_matrix = np.eye(kron_A.shape[0])

initV = pinv(diag_matrix - kron_A) @ Q.flatten(order='F').reshape(-1, 1)
print(initV.shape)
initV = initV.reshape((r * p, r * p), order='F')
print(initV)
print(initV.shape)

### end of the custom Lyapunov method ###




# In[ ]:


import numpy as np
import pandas as pd
from scipy.linalg import eig, pinv, block_diag
from numpy.linalg import inv

def ricSW(standardized_df, q, r, p, start_date, end_date):
    """
    Computes parameters for a factor model using standardized data.

    Parameters:
    standardized_df : pd.DataFrame
        Standardized and balanced panel data (with date as index).
    q : int
        Rank for reduced Q covariance matrix (if applicable).
    r : int
        Number of factors.
    p : int
        Lag order for VAR.
    start_date : str
        Start date for data selection (e.g., '1962-09-01').
    end_date : str
        End date for data selection (e.g., '1999-12-01').

    Returns:
    dict
        A dictionary containing factor model parameters.
    """
    # Filter the data based on the given date range
    standardized_df = data.loc[start_date:end_date]

    # Convert the DataFrame to a NumPy array for numerical operations
    x = standardized_df.values  # Assuming standardized_df is a DataFrame
    T, N = x.shape  # T: number of rows (time periods), N: number of columns (features)
    nlag = p - 1  # Order of lags in the VAR model for the factors. Typically zero if p=1 (number of additional lags beyond t-1)

    # Compute covariance matrix of the data
    cov_x = np.cov(x, rowvar=False)  # Computing the covariance of the data in x. Rowvar=False treats columns as variables

    # Perform eigendecomposition of the covariance matrix
    eigvals, eigvecs = eig(cov_x)
    idx = eigvals.argsort()[::-1]  # Sort eigenvalues in descending order
    eigvals, eigvecs = eigvals[idx][:r], eigvecs[:, idx][:, :r]  # Select the top r eigenvalues and eigenvectors

    # Compute the principal components (factor estimates)
    F = x @ eigvecs  # Transforms the original data x into a new space defined by the eigenvectors

    # Estimate the covariance matrix of the idiosyncratic component (R)
    R = np.diag(np.diag(np.cov(x - F @ eigvecs.T, rowvar=False)))

    # For VAR(1), we just need the first lag: F_{t-1}
    if p == 1:
        Z = F[:-1, :]  # Lagged values: F_{t-1}
        z = F[1:, :]  # Current values: F_t
    else:
        # For VAR(p), we need p lags: F_{t-1}, F_{t-2}, ..., F_{t-p}
        Z = np.hstack([F[p - kk - 1:-(kk + 1), :] for kk in range(p)])  # Stack lags F_{t-1}, F_{t-2}, ..., F_{t-p}
        z = F[p:, :]  # Current values: F_t

    # Estimate the VAR coefficients using OLS
    A_temp = inv(Z.T @ Z) @ Z.T @ z  # Ordinary least squares estimation for VAR(1)
    A = np.zeros((r, r * p))  # Initialize the A matrix
    A[:r, :r * p] = A_temp.T  # Store the estimated coefficients in the A matrix

    # Compute the covariance matrix of the residuals (idiosyncratic errors)
    e = z - Z @ A_temp  # VAR residuals
    H = np.cov(e, rowvar=False)  # Covariance matrix of the residuals

    # If r == q, we directly assign H to Q, otherwise, we reduce the rank of Q
    Q = np.zeros((r * p, r * p))  # Initialize Q as a zero matrix
    if r == q:
        Q[:r, :r] = H  # Use the covariance matrix H if rank r equals q
    else:
        eigvals_H, eigvecs_H = eig(H)  # Eigenvalue decomposition of the residual covariance
        idx = eigvals_H.argsort()[::-1][:q]  # Select the top q eigenvalues
        eigvals_H, eigvecs_H = eigvals_H[idx], eigvecs_H[:, idx]
        Q[:r, :r] = eigvecs_H @ np.diag(eigvals_H) @ eigvecs_H.T  # Update the covariance matrix Q

    # Initialize Kalman filter parameters

    initx = F[0, :]  # Initial state vector based on the first observation
    kron_A = np.kron(A, A)  # Kronecker product of A for Kalman filter initialization
    diag_matrix = np.eye(kron_A.shape[0])
    initV = pinv(diag_matrix - kron_A) @ Q.flatten(order='F').reshape(-1, 1)  # Steady-state covariance of the system
    initV = initV.reshape((r * p, r * p), order='F')  # Reshape to matrix form

    # Create the matrix C for the measurement equation
    C = np.hstack((eigvecs, np.zeros((N, r * nlag))))  # Stack eigenvectors with zeros for lagged terms

    return {
        "A": A, "C": C, "Q": Q, "R": R, "initx": initx, "initV": initV
    }

result = ricSW(data, q=2, r=2, p=1, start_date='1962-09-01', end_date='1999-12-01')

# Print the entire dictionary of results
for key, value in result.items():
    print(f"{key}:")
    print(value)
    print("\n")


# Q represents the covariance matrix of the process noise in the state-space model. It defines the uncertainty in the evolution of the latent factors over time. In state-space models, the latent state (in this case, the factors) evolves according to some dynamics, and Q captures how much uncertainty or randomness there is in the evolution of these factors. Essentially, it measures the "noise" in the factor dynamics.
# 
# Structure: The matrix Q is initialized as a block matrix with the top-left r x r block set to the identity matrix (np.eye(r)), which ensures that the factors have independent unit variance at the start. The rest of the matrix (Q[r:, r:]) is zero. Q is updated after estimating the residuals (the errors between the actual and predicted values of the factors) from the VAR model. This updated Q is typically used to describe the variance of the residuals that cannot be explained by the model.
# 
# Dimensions: Q has size (r * p, r * p), where: r is the number of factors, p is the lag order in the VAR model.
# 
# Interpretation in the model: The diagonal block of Q that corresponds to the factors (Q[:r, :r]) describes the variance of the factors. This tells us how much uncertainty exists in the factor process itself, while the off-diagonal blocks (which are zero in your initialization) would typically describe any cross-covariance terms if factors are related to each other.

# C represents the factor loadings matrix that links the observed data to the underlying factors in the model. In a factor model, the observed variables are typically assumed to be linear combinations of a smaller number of latent (unobserved) factors. The factor loadings matrix C maps the factors to the observed variables.
# 
# Structure: The matrix C is constructed by horizontally stacking the eigenvectors of the covariance matrix of the observed data (eigvecs) and a matrix of zeros. The eigenvectors represent the directions in the data space that explain the most variance. The zeros in the matrix indicate that there are no contributions from lagged values of the factors initially.
# 
# Dimensions: The size of C is (N, r * nlag), where: N is the number of observations (or variables), r is the number of factors, nlag is the number of lags considered (e.g., p - 1 in the model).

# R represents the covariance matrix of the idiosyncratic (residual) errors that are specific to the observed data. In factor models, after explaining the observed variables by the underlying factors, the remaining unexplained part is captured as the idiosyncratic error. This error is specific to each observed variable and cannot be explained by the common factors.
# 
# Structure: The matrix R is computed by first determining the residuals of the factor model. This is done by calculating the covariance between the observed data and the factors and then subtracting the explained part (which is the projection onto the factors). The matrix R reflects the variance (or covariance) of these residuals, which is typically diagonal if the residuals are uncorrelated across variables.
# 
# Dimensions: R has size (N, N) (for the number of observations), but it is typically assumed to be diagonal in many factor models, where the diagonal elements represent the variance of the idiosyncratic error for each observed variable.
# 
# Interpretation in the model: R captures the noise or error in the observed data that is not explained by the common factors. If the residuals are small (low R values), it means the factors explain most of the variance in the observed data, while larger values of R indicate that a significant portion of the data’s variance remains unexplained by the factors.

# initx represents the initial values of the latent factors (the unobserved components) at time t=0. In the state-space formulation of your factor model, the factors evolve over time according to a dynamic process (often a VAR model, as in your case). The vector initx is the starting point of the latent factors, reflecting their values at the beginning of the process.
# 
# Structure: initx is set to the first row of the factor matrix F, i.e., F[0, :]. F is the matrix of principal components (factors) that are computed by projecting the standardized data onto the eigenvectors of the data’s covariance matrix.
# 
# Therefore, initx is essentially the first set of values for the factors based on the data at the start of the series.
# 
# Dimensions: initx is a vector with dimensions (r,), where r is the number of factors in your model. It contains the initial state of the latent factors.
# 
# Role in the Model: initx provides the starting point for the Kalman filter or any other state estimation process you're using. This initial state vector is needed for making predictions and updating the state of the model as new data comes in. It determines the "initial belief" about the value of the latent factors before observing any new data.

# initV represents the initial covariance matrix of the latent factors' state vector at the start of the time series. It encapsulates the uncertainty about the initial state of the factors. This matrix tells us how much variance there is in the initial values of the factors and how they might be correlated with each other.
# 
# Structure: initV is derived from the Kronecker product of the matrix A (the transition matrix from the VAR model) and the identity matrix, then adjusted using the covariance matrix Q. In the line initV = pinv(diag_matrix - kron_A) @ Q.flatten(order='F').reshape(-1, 1), the initial covariance matrix is calculated through a process involving the transition matrix A, the covariance matrix Q, and the inverse of the transformation operator (represented by the Kronecker product). pinv(diag_matrix - kron_A) uses the Moore-Penrose pseudoinverse to handle any potential singularities in the matrix and compute a stable estimate of the covariance structure.
# 
# Dimensions: initV has dimensions (r * p, r * p), where: r is the number of factors (latent variables), p is the lag order of the VAR model. r * p reflects the size of the state vector that incorporates both the factors and their lags.
# 
# Role in the Model: initV is crucial for filtering and smoothing in state-space models (such as the Kalman filter). It reflects the uncertainty about the initial state of the system and is used to update beliefs about the factors as new data arrives. Specifically, it describes how the errors or shocks to the system propagate and evolve over time.

# In[ ]:


#### Comparison using packages ####

q = 2  # Rank for reduced Q covariance matrix
r = 2  # Number of factors
p = 1  # Lag order for VAR (1)
start_date = '1962-09-01'
end_date = '1999-12-01'

### Estimation Using statsmodels (for VAR) ###
from statsmodels.tsa.api import VAR

# Select the data from the relevant date range
df_subset = data.loc[start_date:end_date]

# Fit a VAR model to the data
model = VAR(df_subset)
var_results = model.fit(1)  # Fit with 1 lag (p=1)

# Extract the VAR coefficients and residual covariance
A_statsmodels = var_results.coefs[0].T  # Coefficients for the VAR(1) model
H = var_results.sigma_u  # Residual covariance matrix from VAR model

print(A_statsmodels)
print(H)

### Estimation Using Kalman Filter ###

### Do not execute! #################

######################################

###
get_ipython().system('pip install pykalman')

from pykalman import KalmanFilter
import numpy as np

# Define the number of factors (latent states) r
r = 2  # This could be set based on your model specification

# Initialize Kalman Filter
kf = KalmanFilter(
    transition_matrices=A_statsmodels,  # VAR coefficients as state transition matrix
    observation_matrices=np.eye(df_subset.shape[1]),  # Assuming identity observation matrix (for simplicity)
    initial_state_mean=np.zeros(r),  # Initial state vector (zeros or first observed values)
    initial_state_covariance=np.eye(r) * 1e-2,  # Initial state covariance (small diagonal matrix)
    em_vars=['transition_covariance', 'observation_covariance']  # Allow Kalman Filter to estimate covariance matrices
)

# Estimate the latent factors (state means) and covariance (state covariances)
state_means, state_covariances = kf.filter(df_subset.values)  # Pass the observed data (df_subset.values)

# The latent factors (F) are the state estimates
F_kf = state_means  # These are the estimated latent factors

# Print the estimated latent factors
print("Estimated Latent Factors (F_kf):")
print(F_kf)

# If needed, you can also extract the Kalman gain and other details


# In[ ]:


### Comparison using libraries ###

## Do not run! ##

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Assuming 'data' is your DataFrame and contains the time series data with a DateTime index
# Filter data between '1962-09-01' and '1999-12-01'
data_filtered = data.loc['1962-09-01':'1999-12-01']

# Perform PCA to extract the first r principal components
r = 2  # Number of components (you can adjust this as needed)
pca = PCA(n_components=r)
pca.fit(data_filtered)

# Get the first r principal components (F)
F = pca.transform(data_filtered)

# Get the factor loadings (the eigenvectors)
factor_loadings = pca.components_.T  # Shape: (n_features, r)

# Get the covariance matrix of the residuals (idiosyncratic component, R)
# First, reconstruct the data from the principal components
reconstructed_data = F @ factor_loadings.T
residuals = data_filtered - reconstructed_data  # Residuals (idiosyncratic component)

# Covariance matrix of the residuals (R)
R = np.cov(residuals, rowvar=False)

# Print results
print("Factor Loadings (Eigenvectors):")
print(factor_loadings)
print("\nCovariance Matrix of the Idiosyncratic Component (R):")
print(R)

# Optionally, if you want the variance explained by the factors
explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance (Proportion of variance explained by each component):")
print(explained_variance)


# In[ ]:


### Comparison using libraries ###

## Do not run! ##

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR

# Assuming 'data' is the standardized dataset filtered for the desired period
data_filtered = data.loc['1962-09-01':'1999-12-01']

# Step 1: Perform PCA to extract the first r principal components
r = 2  # Number of factors
pca = PCA(n_components=r)
F = pca.fit_transform(data_filtered)  # F is the matrix of factors (T x r)

# Convert F into a DataFrame for easier manipulation in statsmodels
F_df = pd.DataFrame(F, index=data_filtered.index, columns=[f"Factor_{i+1}" for i in range(r)])

# Step 2: Fit a VAR model to the extracted factors
var_model = VAR(F_df)
var_result = var_model.fit(maxlags=1)  # Assuming a VAR(1) structure

# Extract the autoregressive coefficient matrices (A)
A = var_result.coefs  # Shape: (lags, r, r)
# A[0] is the matrix of coefficients for the VAR(1) model

# Extract the covariance matrix of residuals (H)
H = var_result.sigma_u  # Covariance matrix of residuals

# Step 3: Print the results
print("Autoregressive Coefficient Matrix (A):")
print(A[0])  # A[0] corresponds to the VAR(1) coefficient matrix

print("\nCovariance Matrix of Residuals (H):")
print(H)

# Step 4: Optional - Explained Variance of the Principal Components
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance (Proportion of variance explained by each factor):")
print(explained_variance)


# In[ ]:


## Do not run! ##

## Kalman Filter (Forward Pass) Only:

import numpy as np
from numpy.linalg import pinv, det

# Assume the necessary matrices are already defined: y, A, C, Q, R, init_x, init_V, model

# Step 1: Define the vintage cutoff date
vintage_date = '1999-12-01'

# Step 2: Filter data for rows strictly after the vintage date
data_after_vintage = data.loc[vintage_date:]

# Step 3: Drop the vintage date row itself (if necessary)
data_after_vintage = data_after_vintage.iloc[1:]

# Print the resulting DataFrame
print("Data After Vintages:")
print(data_after_vintage.shape)


os = data_after_vintage.shape[1]  # Number of observed variables
T = data_after_vintage.shape[0]   # Number of time steps (rows in y)
ss = A.shape[0]  # State space size

# Initialize filtered state and covariance arrays
x = np.zeros((ss, T))          # Filtered state estimates
V = np.zeros((ss, ss, T))      # State covariance matrices
VV = np.zeros((ss, ss, T))     # Cross-covariance matrices
loglik = 0                     # Log-likelihood accumulator

T = data_after_vintage.shape[0]
print(T)
AA = np.repeat(A[:, :, np.newaxis], T, axis=2)
QQ = np.repeat(Q[:, :, np.newaxis], T, axis=2)
CC = np.repeat(C[:, :, np.newaxis], T, axis=2)
RR = np.repeat(R[:, :, np.newaxis], T, axis=2)
model = range(1, T+1)
print(initV.shape)

print(initx.shape)
print(A.shape)

# Forward pass: Run the Kalman filter
for t in range(T):
    m = model[t]

    if t == 0:
        prevx = initx # (2, ) 1-dimension array with two elements
        prevV = initV # 2 by 2 matrix
        initial = True
    else:
        prevx = x[:, t-1].reshape(-1, 1)
        prevV = V[:, :, t-1]
        initial = False

    # Prediction step
    if initial:
        xpred = prevx
        Vpred = prevV
    else:
        xpred = A @ prevx
        Vpred = A @ prevV @ A.T + Q[:, :, t]

    # Innovation
    e = data_after_vintage[:, t].reshape(-1, 1) - C[:, :, t] @ xpred
    S = C[:, :, t] @ Vpred @ C[:, :, t].T + R[:, :, t]
    Sinv = np.linalg.inv(S)

    # Log-likelihood calculation
    detS = det(S)
    loglik_step = -0.5 * (np.log(detS) + e.T @ Sinv @ e + len(e) * np.log(2 * np.pi))

    # Kalman gain
    K = Vpred @ C[:, :, t].T @ Sinv

    # State and covariance update
    x[:, t] = (xpred + K @ e).flatten()
    V[:, :, t] = (np.eye(ss) - K @ C[:, :, t]) @ Vpred
    VV[:, :, t] = (np.eye(ss) - K @ C[:, :, t]) @ A[:, :, m-1] @ Vpred

    # Print current step outputs for verification
    print(f"Step {t + 1} - Kalman Filter Output")
    print(f"x: {x[:, t]}")
    print(f"V: {V[:, :, t]}")
    print(f"Log-likelihood: {loglik_step.item()}")
    loglik += loglik_step.item()

# After the forward pass, print the total log-likelihood
print(f"Total Log-Likelihood after Forward Pass: {loglik}")


# In[ ]:


##############################
### Restart execution here ###
##############################

## Kalman filter (forward pass) ##

import numpy as np

# Assumed data
vintage_date = '1999-12-01'

# Step 1: Define the vintage cutoff date
data_after_vintage = data.loc[vintage_date:]

# Step 2: Drop the vintage date row itself
data_after_vintage = data_after_vintage.iloc[1:]
obs_data = data_after_vintage.values  # Dataset after the vintage

# # Assumed parameters (from DFM estimation)

print(A)
print(V)
print(C)
print(R)
print(A.shape)
print(V.shape)
print(C.shape)
print(R.shape)
# A = np.array([[0.8, 0.1], [0.1, 0.7]])  # VAR(1) coefficient matrix
# V = np.array([[0.05, 0.0], [0.0, 0.05]])  # Process noise covariance matrix (replaces Q)
# C = eigvecs  # Factor loadings matrix (from DFM estimation)
# R = np.diag(np.diag(np.cov(x - F @ C.T, rowvar=False)))  # Covariance of idiosyncratic errors
n_obs, n_factors = obs_data.shape[0], A.shape[0]

# Initialization
print(initx)
print(V)
x_0 = F[-1, :]  # Initial latent factors
print(x_0)
P_0 = V  # Steady-state covariance

# Prepare storage
F_estimates = []  # To store factor estimates
P_estimates = []  # To store covariance estimates

# Initialize
F_t = x_0
P_t = P_0

# Forward pass
for t in range(n_obs):
    # Observation at current time step
    x_t = obs_data[t, :]

    # Prediction step
    F_pred = A @ F_t  # Predicted state
    P_pred = A @ P_t @ A.T + V  # Predicted covariance (using V)

    # Update step
    K_t = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)  # Kalman gain
    F_t = F_pred + K_t @ (x_t - C @ F_pred)  # Updated state estimate
    P_t = (np.eye(n_factors) - K_t @ C) @ P_pred  # Updated covariance estimate

    # Store results
    F_estimates.append(F_t)
    P_estimates.append(P_t)

# Convert results to arrays
F_estimates = np.array(F_estimates)
P_estimates = np.array(P_estimates)

print("Latent Factor Estimates:")
print(F_estimates) # F_estimates: A matrix where each row contains the estimated latent factors for each time step after the vintage date.
print("State Covariance Estimates:")
print(P_estimates) # P_estimates: A matrix containing the covariance estimates of the latent factors for each time step.


# In[ ]:


## Implementing a Kalman smoother ##

# Assuming Kalman filter results: F_estimates (latent factors), P_estimates (covariance matrices)

n_timesteps = F_estimates.shape[0]
n_factors = F_estimates.shape[1]

# Initialize smoother results
F_smoothed = np.zeros_like(F_estimates)
P_smoothed = np.zeros_like(P_estimates)

# Start with the last Kalman filter estimates
F_smoothed[-1] = F_estimates[-1]
P_smoothed[-1] = P_estimates[-1]

# Backward smoothing pass
for t in range(n_timesteps - 2, -1, -1):  # Loop from T-1 to 0
    # Smoother gain
    P_t = P_estimates[t]
    P_t1_pred = A @ P_t @ A.T + V
    J_t = P_t @ A.T @ np.linalg.inv(P_t1_pred)

    # Update smoothed state
    F_smoothed[t] = F_estimates[t] + J_t @ (F_smoothed[t + 1] - A @ F_estimates[t])

    # Update smoothed covariance
    P_smoothed[t] = P_t + J_t @ (P_smoothed[t + 1] - P_t1_pred) @ J_t.T

# Results
print("Smoothed latent factors (F_smoothed):")
print(F_smoothed)
print("Smoothed covariances (P_smoothed):")
print(P_smoothed)


# In[ ]:


### Alternative implementation of the Kalman smoother (Real-Time Fixed-Lag Smoothing)

# Kalman filter estimates (real-time)
F_filtered = []  # Real-time factor estimates
P_filtered = []  # Real-time covariance estimates
L = 3

# Real-time data stream (assume x_data is your observed data matrix)
for t in range(n_obs):
    # Kalman filter forward pass
    F_t = A @ F_t_prev
    P_t = A @ P_t_prev @ A.T + V

    # Update with observation at time t
    K_t = P_t @ C.T @ np.linalg.inv(C @ P_t @ C.T + R)
    F_t = F_t + K_t @ (obs_data[t] - C @ F_t)
    P_t = P_t - K_t @ C @ P_t

    # Store estimates
    F_filtered.append(F_t)
    P_filtered.append(P_t)

    # Perform fixed-lag smoothing
    if t >= L:  # Smoothing possible after reaching lag size
        for tau in range(t - L, t + 1):
            J_tau = P_filtered[tau] @ A.T @ np.linalg.inv(P_filtered[tau + 1])
            F_filtered[tau] = F_filtered[tau] + J_tau @ (F_filtered[tau + 1] - A @ F_filtered[tau])
            P_filtered[tau] = P_filtered[tau] + J_tau @ (P_filtered[tau + 1] - P_filtered[tau + 1]) @ J_tau.T

    # Store previous state for next iteration
    F_t_prev = F_t
    P_t_prev = P_t

# Convert lists to arrays
F_filtered = np.array(F_filtered)
P_filtered = np.array(P_filtered)

# Nowcast using the smoothed factors
print("Real-time smoothed factors for nowcasting:", F_filtered[-L:])


# In[ ]:


# Kalman filter and smoother
# Kalman filter and smoother initialization
# print(T)
# print(N)

# Kalman filter initialization
F_t_prev = initx  # Initial latent factors
P_t_prev = V  # Initial covariance matrix

F_filtered = []  # Store real-time filtered estimates
P_filtered = []  # Store covariance matrices
F_smoothed = []  # Store smoothed estimates

T2 = len(obs_data)

# Define a fixed lag (3 months for quarterly lag)
L = 3  # Use 3 months for quarterly smoothing

# print(obs_data)

# Real-Time Fixed-Lag Smoothing (Quarterly)
for t in range(T2):
    # Kalman Filter Forward Pass
    # Predict step
    F_t_pred = A @ F_t_prev  # Predict latent factors
    P_t_pred = A @ P_t_prev @ A.T + V  # Predict covariance

    # Check if t is within the bounds of obs_data
    if t < len(obs_data):
        # Update step with observation obs_data[t]
        K_t = P_t_pred @ C.T @ np.linalg.inv(C @ P_t_pred @ C.T + R)  # Kalman gain
        F_t = F_t_pred + K_t @ (obs_data[t] - C @ F_t_pred)  # Update factors
        P_t = P_t_pred - K_t @ C @ P_t_pred  # Update covariance

    # Store filtered estimates
    F_filtered.append(F_t)
    P_filtered.append(P_t)

    # Real-Time Fixed-Lag Smoothing (quarterly smoothing)
    # Perform smoothing after each quarter (March, June, September, December)
    if (t + 1) % 3 == 0:  # Check if t is the end of a quarter (March, June, September, December)
        for tau in range(max(0, t - L + 1), t + 1):  # Ensure valid range
            if tau + 1 < len(F_filtered):
                # Smoothing formula: Use all earlier estimates (up to the end of the current quarter)
                J_tau = P_filtered[tau] @ A.T @ np.linalg.inv(P_t_pred)
                F_filtered[tau] = F_filtered[tau] + J_tau @ (F_filtered[tau + 1] - A @ F_filtered[tau])
                P_filtered[tau] = P_filtered[tau] + J_tau @ (P_filtered[tau + 1] - P_t_pred) @ J_tau.T
            else:
                print(f"Skipping tau: {tau}, t: {t} due to index out of range.")

    # Store smoothed estimates for debugging or further analysis
    F_smoothed.append(F_filtered[t])

    # Prepare for next iteration
    F_t_prev = F_t
    P_t_prev = P_t

# Convert lists to arrays for further analysis
F_filtered = np.array(F_filtered)
P_filtered = np.array(P_filtered)
F_smoothed = np.array(F_smoothed)

# Print smoothed factors
print("Filtered latent factors (F_filtered):", F_filtered.shape)
print("Smoothed latent factors (F_smoothed):", F_smoothed.shape)

print(F_filtered)
print(F_smoothed)

