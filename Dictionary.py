import pickle
import numpy as np
import pandas as pd

# Load the Kalman output data
with open('shared_variables.pkl', 'rb') as f:
    kalman_data = pickle.load(f)

# Extract the variables
filtered_factors = kalman_data['F_filtered']
smoothed_factors = kalman_data['F_smoothed']
filtered_covariances = kalman_data['P_filtered']
observations = kalman_data['obs_data']  # Include this if needed

# Define parameters for the dictionary method
start_date = "2000-01-01"  # Adjust to your context
frequency = "M"  # Monthly frequency

############### Method proper #####################

def associate_factors_with_time_index(filtered_factors, smoothed_factors, observations, start_date, frequency):
    """
    Associates filtered and smoothed latent factors with their corresponding time index.

    Parameters:
        filtered_factors (np.ndarray): The filtered latent factors (shape: T x r).
        smoothed_factors (np.ndarray): The smoothed latent factors (shape: T x r).
        observations (np.ndarray): Observational data (to determine length and context).
        start_date (str): Start date of the observation period (e.g., "2000-01-01").
        frequency (str): Frequency of the data (e.g., 'M' for monthly, 'Q' for quarterly).

    Returns:
        dict: A dictionary with DataFrames for filtered and smoothed factors, and quarterly smoothed values.
    """
    # Generate a time index based on the start date and frequency
    T = len(observations)
    time_index = pd.date_range(start=start_date, periods=T, freq=frequency)

    # Create DataFrames for filtered and smoothed factors
    filtered_df = pd.DataFrame(filtered_factors, index=time_index,
                               columns=[f"Factor_{i + 1}" for i in range(filtered_factors.shape[1])])
    smoothed_df = pd.DataFrame(smoothed_factors, index=time_index,
                               columns=[f"Factor_{i + 1}" for i in range(smoothed_factors.shape[1])])

    # Identify end-of-quarter dates for quarterly smoothed factors
    if frequency == 'M':  # Monthly data
        quarter_end_dates = time_index[time_index.to_series().dt.month.isin([3, 6, 9, 12])]
    elif frequency == 'Q':  # Quarterly data
        quarter_end_dates = time_index  # All dates are quarter-end
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Expected 'M' or 'Q'.")

    # Extract smoothed factors for quarter-end dates
    quarterly_smoothed_df = smoothed_df.loc[quarter_end_dates]

    # Return results as a dictionary
    return {
        "filtered_factors": filtered_df,
        "smoothed_factors": smoothed_df,
        "quarterly_smoothed_factors": quarterly_smoothed_df
    }

# Call the method
results = associate_factors_with_time_index(filtered_factors, smoothed_factors, observations, start_date, frequency)

# Access the results
filtered_df = results["filtered_factors"]
smoothed_df = results["smoothed_factors"]
quarterly_smoothed_df = results["quarterly_smoothed_factors"]

# Print the outputs
print("Filtered Factors:")
print(filtered_df)

print("\nSmoothed Factors:")
print(smoothed_df)

print("\nQuarterly Smoothed Factors:")
print(quarterly_smoothed_df)
