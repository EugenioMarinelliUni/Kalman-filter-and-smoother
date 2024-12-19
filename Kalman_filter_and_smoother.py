# %%
import pickle
import numpy as np
import pandas as pd

# Open the pickle file
with open('shared_variables.pkl', 'rb') as file:
    # Load the content of the pickle file
    data = pickle.load(file)

# Access the variables using their keys
A = data.get('A')
C = data.get('C')
R = data.get('R')
V = data.get('V')
all_data_numpy_array = data.get('all_data_numpy_array')
initx = data.get('initx')
P_final = data.get('P_final')
F_smoothed = data.get('F_smoothed')

# Print or work with the variables
print(A, C, R, all_data_numpy_array, initx, P_final, F_smoothed, V)
print(type(all_data_numpy_array))
print(all_data_numpy_array)

print(type(V))
#######################################
file_path = 'standardized_compact_dataset.csv'
vintage_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
print(vintage_data.head())
print(vintage_data)
######################################

# Initialize global state variables for the first iteration
F_t_prev_global = initx  # Initial latent factors (2x1 matrix)
P_t_prev_global = P_final  # Initial covariance matrix (2x2)

smoothed_factors_dict = {}

# Define the date range for iteration
start_date = "2000-01"  # January 2000

last_date = vintage_data.index[-1]

# Convert the last date to "YYYY-MM" format
end_date = last_date.strftime("%Y-%m")

# Print the result
print("End date set to:", end_date)

date_range = pd.date_range(start=start_date, end=end_date, freq="MS")  # Monthly intervals
print(date_range)
# Iterate over the date range
# Kalman Filtering and Smoothing Iteration

####

#### Improved method, returns a dictionary of dictionaries. ###

# Initialize an empty DataFrame to store all smoothed factors
smoothed_factors_df = pd.DataFrame(columns=["Global_Date", "Factor_Date", "Factor_Values"])

# For each month in the date range
for current_date in date_range:
    # Convert the current_date into an index range for vintage_data
    end_index = vintage_data.index.get_loc(current_date)
    vintage_data_subset = vintage_data.iloc[: end_index + 1]  # Data up to current_date

    # Initialize for the current iteration
    F_t_prev = F_t_prev_global  # Use global estimates from the previous iteration
    P_t_prev = P_t_prev_global  # Use global covariance from the previous iteration
    F_filtered = []  # Store filtered latent factors
    P_filtered = []  # Store covariance matrices
    P_pred_filtered = []

    # Step 1: Kalman Filtering on Vintage Data
    for t in range(1, vintage_data_subset.shape[0]):  # Start from t=1 to skip initial guess
        # Prediction step: Calculate predicted latent factors and covariance
        F_t_pred = np.dot(A, F_t_prev).reshape(-1, 1)  # Ensure F_t_pred is a (2, 1) column vector
        P_t_pred = np.dot(np.dot(A, P_t_prev), A.T) + V

        # Innovation (difference between actual data and prediction)
        innovation = vintage_data_subset.iloc[t, :].values.reshape(-1, 1) - np.dot(C, F_t_pred)

        # Calculate Kalman Gain
        K_t = np.dot(np.dot(P_t_pred, C.T), np.linalg.inv(np.dot(np.dot(C, P_t_pred), C.T) + R))

        # Update step: Correct the prediction with the innovation
        F_t = F_t_pred + np.dot(K_t, innovation)  # Shape: (2, 1) + (2, 1) -> (2, 1)

        # Update covariance
        P_t = P_t_pred - np.dot(np.dot(K_t, C), P_t_pred)

        # Store the filtered results
        F_filtered.append(F_t)
        P_filtered.append(P_t)
        P_pred_filtered.append(P_t_pred)

        # Update the previous values for the next iteration
        F_t_prev = F_t
        P_t_prev = P_t

    # Step 2: Kalman Smoothing
    F_smoothed = F_filtered.copy()
    P_smoothed = P_filtered.copy()

    for t in reversed(range(len(F_filtered) - 1)):
        # Compute the smoothing gain (J_t) for time step t
        P_t_pred = P_pred_filtered[t]
        J_t = np.dot(np.dot(P_filtered[t], A.T), np.linalg.inv(P_t_pred))

        # Perform smoothing for factors
        F_smoothed[t] = F_filtered[t] + np.dot(J_t, (F_smoothed[t + 1] - np.dot(A, F_filtered[t])))

        # Perform smoothing for covariance
        P_smoothed[t] = P_filtered[t] + np.dot(np.dot(J_t, (P_smoothed[t + 1] - P_t_pred)), J_t.T)

    # Convert smoothed factors to array
    F_smoothed_array = np.array(
        F_smoothed).squeeze()  # Convert list to numpy array and remove single-dimensional entries

    # Create a dictionary to store smoothed factors with dates
    smoothed_factors_dict[current_date] = {}

    # Store the smoothed factors with their corresponding dates
    for i, factor in enumerate(F_smoothed_array):
        smoothed_factors_dict[current_date][vintage_data_subset.index[i]] = factor

    # Store the smoothed factors with their corresponding dates in the DataFrame
    for i, factor in enumerate(F_smoothed_array):
        smoothed_factors_df = pd.concat([
            smoothed_factors_df,
            pd.DataFrame({
                "Global_Date": [current_date],
                "Factor_Date": [vintage_data_subset.index[i]],
                "Factor_Values": [factor]
            })
        ], ignore_index=True)

    # Update global state variables for the next iteration
    F_t_prev_global = F_t  # Final latent factors from this iteration
    P_t_prev_global = P_t  # Final covariance matrix from this iteration

    # Print the result at the end of each iteration
    print(f"Smoothed Factors for {current_date}:")
    for date, factor in smoothed_factors_dict[current_date].items():
        print(f"{date}: {factor}")
    print("-" * 50)  # Separator for readability

# Save the DataFrame to a CSV file
smoothed_factors_df.to_csv("smoothed_factors.csv", index=False)
print("Smoothed factors saved to smoothed_factors.csv.")

############### THIS METHOD WORKS!!! ###################à
# # For each month in the date range
# for current_date in date_range:
#     # Convert the current_date into an index range for vintage_data
#     end_index = vintage_data.index.get_loc(current_date)
#     vintage_data_subset = vintage_data.iloc[: end_index + 1]  # Data up to current_date
#
#     # Initialize for the current iteration
#     F_t_prev = F_t_prev_global  # Use global estimates from the previous iteration
#     P_t_prev = P_t_prev_global  # Use global covariance from the previous iteration
#     F_filtered = []  # Store filtered latent factors
#     P_filtered = []  # Store covariance matrices
#     P_pred_filtered = []
#
#     # Step 1: Kalman Filtering on Vintage Data
#     for t in range(1, vintage_data_subset.shape[0]):  # Start from t=1 to skip initial guess
#         # Prediction step: Calculate predicted latent factors and covariance
#         F_t_pred = np.dot(A, F_t_prev).reshape(-1, 1)  # Ensure F_t_pred is a (2, 1) column vector
#         P_t_pred = np.dot(np.dot(A, P_t_prev), A.T) + V
#
#         # Innovation (difference between actual data and prediction)
#         innovation = vintage_data_subset.iloc[t, :].values.reshape(-1, 1) - np.dot(C, F_t_pred)
#
#         # Calculate Kalman Gain
#         K_t = np.dot(np.dot(P_t_pred, C.T), np.linalg.inv(np.dot(np.dot(C, P_t_pred), C.T) + R))
#
#         # Update step: Correct the prediction with the innovation
#         F_t = F_t_pred + np.dot(K_t, innovation)  # Shape: (2, 1) + (2, 1) -> (2, 1)
#
#         # Update covariance
#         P_t = P_t_pred - np.dot(np.dot(K_t, C), P_t_pred)
#
#         # Store the filtered results
#         F_filtered.append(F_t)
#         P_filtered.append(P_t)
#         P_pred_filtered.append(P_t_pred)
#
#         # Update the previous values for the next iteration
#         F_t_prev = F_t
#         P_t_prev = P_t
#
#     # Step 2: Kalman Smoothing
#     F_smoothed = F_filtered.copy()
#     P_smoothed = P_filtered.copy()
#
#     for t in reversed(range(len(F_filtered) - 1)):
#         # Compute the smoothing gain (J_t) for time step t
#         P_t_pred = P_pred_filtered[t]
#         J_t = np.dot(np.dot(P_filtered[t], A.T), np.linalg.inv(P_t_pred))
#
#         # Perform smoothing for factors
#         F_smoothed[t] = F_filtered[t] + np.dot(J_t, (F_smoothed[t + 1] - np.dot(A, F_filtered[t])))
#
#         # Perform smoothing for covariance
#         P_smoothed[t] = P_filtered[t] + np.dot(np.dot(J_t, (P_smoothed[t + 1] - P_t_pred)), J_t.T)
#
#     # Convert smoothed factors to array and store them
#     F_smoothed_array = np.array(F_smoothed).squeeze()  # Convert list to numpy array and remove single-dimensional entries
#     smoothed_factors_dict[current_date] = F_smoothed_array  # Store the smoothed factors for the current date
#
#     # Update global state variables for the next iteration
#     F_t_prev_global = F_t  # Final latent factors from this iteration
#     P_t_prev_global = P_t  # Final covariance matrix from this iteration
#
#     # Print the result at the end of each iteration
#     print(f"Smoothed Factors for {current_date}:")
#     print(F_smoothed_array)  # Print the smoothed factors for the current date
#     print("-" * 50)  # Separator for readability


######################################






#################################

###################################

import pickle
import os

# Filepath for the pickle file
pickle_file = 'shared_variables.pkl'

# # Data to save
# new_data = {
#     'F_filtered': F_filtered,
#     'F_smoothed': F_smoothed,
#     'P_filtered': P_filtered,
#     'obs_data': obs_data,  # Only save if not already in the file
# }
#
# # Check if the file exists
# if os.path.exists(pickle_file):
#     # Load existing data
#     with open(pickle_file, 'rb') as f:
#         existing_data = pickle.load(f)
# else:
#     # Initialize an empty dictionary if the file doesn't exist
#     existing_data = {}
#
# # Update the existing data with new entries
# for key, value in new_data.items():
#     if key in existing_data:
#         print(f"{key} already exists in the pickle file. Skipping save for this key.")
#     else:
#         existing_data[key] = value
#
# # Save the updated dictionary back to the pickle file
# with open(pickle_file, 'wb') as f:
#     pickle.dump(existing_data, f)
#
# print("Data saved to shared_variables.pkl")
