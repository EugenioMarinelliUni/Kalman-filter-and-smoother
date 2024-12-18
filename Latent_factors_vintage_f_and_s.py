### Filter and Smooth the Vintage Data ###

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle


# Load the variables from the pickle file
with open('shared_variables_vintage.pkl', 'rb') as f:
    shared_data = pickle.load(f)

# Access the numpy arrays and the DataFrame
A = shared_data['A']
C = shared_data['C']
R = shared_data['R']
V = shared_data['V']
F = shared_data['F']
filtered_data = shared_data['filtered_data']  # This will be a DataFrame
vintage_data= filtered_data.values

# Check if the DataFrame and arrays have been loaded correctly
print("A:", A)
print("C:", C)
print("R:", R)
print("V:", V)
print("F:", F)
print("filtered_data:\n", vintage_data)

# Print the data types of the variables

print("Data type of A:", type(A))
print("Data type of C:", type(C))
print("Data type of R:", type(R))
print("Data type of V:", type(V))
print("Data type of F:", type(F))
print("Data type of filtered_data:", type(vintage_data))

print(vintage_data.shape)
print(F.shape)

###

# Initializing Kalman filter variables
F_t_prev = np.zeros((2, 1))  # Initial latent factors (2 factors, 1 column: 2x1 matrix)
P_t_prev = np.eye(2)  # Initial covariance matrix (2x2)

F_filtered = []  # Store filtered latent factors
P_filtered = []  # Store covariance matrices
P_pred_filtered = []

# Step 1: Kalman Filtering on Vintage Data (Apply Kalman filter to latent factors F)
for t in range(1, vintage_data.shape[0]):  # Start from t=1 because t-1 should be available
    # Prediction step (Apply state transition matrix A to previous latent factors F(t-1))
    F_t_pred = np.dot(A, F_t_prev)  # Predicted latent factor at time t (shape: 2x1)
    P_t_pred = np.dot(np.dot(A, P_t_prev), A.T) + V  # Predicted covariance matrix

    # Update step (Kalman Filter update based on observations)
    innovation = vintage_data[t, :].reshape(-1, 1) - np.dot(C, F_t_pred)  # Observation residual (118x1)

    # Step 1: Compute the Kalman Gain K_t
    K_t = np.dot(np.dot(P_t_pred, C.T), np.linalg.inv(np.dot(np.dot(C, P_t_pred), C.T) + R))  # Kalman gain (2x118)

    # Step 2: Update the latent factors F_t
    F_t = F_t_pred + np.dot(K_t, innovation)  # Updated latent factors (shape: 2x1)

    # Step 3: Update the covariance matrix P_t
    P_t = P_t_pred - np.dot(np.dot(K_t, C), P_t_pred)  # Updated covariance matrix (2x2)

    # Store filtered results
    F_filtered.append(F_t)  # Store as 2x1
    P_filtered.append(P_t)  # Store as 2x2
    P_pred_filtered.append(P_t_pred)  # Store predicted covariance matrices for backward pass

    # Update for next iteration
    F_t_prev = F_t  # Update previous latent factors
    P_t_prev = P_t  # Update previous covariance matrix

# Step 2: Perform "once and for all" backward smoothing for factors and covariance matrix
F_smoothed = F_filtered.copy()  # Copy filtered latent factors for smoothing
P_smoothed = P_filtered.copy()  # Copy filtered covariance matrices for smoothing

# Perform backward pass for Kalman smoothing
for t in reversed(range(len(F_filtered) - 1)):  # Start from the second to last step
    # Retrieve predicted covariance matrix for the next step
    P_t_pred = P_pred_filtered[t]  # Use the stored predicted covariance matrix

    # Calculate Kalman smoother gain J_t for factors
    J_t = np.dot(np.dot(P_filtered[t], A.T), np.linalg.inv(P_t_pred))  # Kalman smoother gain (2x2)

    # Apply Kalman smoother formula to adjust the filtered factors
    F_smoothed[t] = F_filtered[t] + np.dot(J_t, (F_smoothed[t + 1] - np.dot(A, F_filtered[t])))  # Adjusted smoothed factors (2x1)

    # Adjust the smoothed covariance matrix
    P_smoothed[t] = P_filtered[t] + np.dot(np.dot(J_t, (P_smoothed[t + 1] - P_t_pred)), J_t.T)  # Adjusted smoothed covariance (2x2)

# Convert to arrays for further use (ensure the correct shape)
F_smoothed = np.array(F_smoothed).squeeze()  # Ensure correct shape (time steps x 2)
P_smoothed = np.array(P_smoothed)  # Ensure correct shape for covariance matrices (time steps x 2 x 2)

print("Smoothed Factors (Vintage 1960-2000):", F_smoothed.shape)
print("Smoothed Covariance Matrices:", P_smoothed.shape)

# Step 3: Use the final smoothed values at the end of 2000 as initial conditions for later estimation
initx = F_smoothed[-1]  # Final smoothed latent factor (2x1 vector)
P_final = P_smoothed[-1]  # Final smoothed covariance matrix (2x2)



# Print results
print("Initial state (initx) from smoothed factors:", initx)
print("Final covariance matrix (P_final) from smoothed factors:", P_final)

######

## Saving the output of interest in a shared file to ensure its reusability in the Kalman_filter_and_smoother file
import pickle

try:
    with open('shared_variables.pkl', 'rb') as file:
        existing_data = pickle.load(file)
except FileNotFoundError:
    existing_data = {}

# Append new data from File 2 to the existing data
existing_data.update({'initx': initx, 'P_final': P_final, 'F_smoothed': F_smoothed, 'V': V})

# Check and print the data types of the variables
# Save everything back to the pickle file (overwrite)
with open('shared_variables.pkl', 'wb') as file:
    pickle.dump(existing_data, file)

print("Variables from intermediate method saved to shared_variables.pkl")

#################

# Test loading the pickle file
with open('shared_variables.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Print to confirm the variables were loaded correctly
print(loaded_data['V'])  # Should print the value of V
print(loaded_data['initx'])  # Should print the value of initx




