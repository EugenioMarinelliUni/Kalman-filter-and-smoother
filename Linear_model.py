import numpy as np
import pandas as pd
import statsmodels.api as sm

# Placeholder for your data
# gdp_growth: pandas Series of QoQ GDP growth rates indexed by quarters
# quarterly_factors: pandas DataFrame of smoothed latent factors indexed by quarters

## Loading the quarterly gdp growth rates from google drave and saving them in the project folder ##

import gdown
import os
import pandas as pd

# Step 1: Define the Google Drive file link or File ID
file_id = '1-3B2AVroNRXytpJ7XOIskhTuuH6zyh6J'  # Replace with the actual File ID
output_file_name = 'GDP_Percent_Change.csv'  # Name to save the file as

# Step 2: Construct the Google Drive download URL
gdrive_url = f'https://drive.google.com/uc?id={file_id}'

# Step 3: Set the path to save the file in the project folder
project_folder = os.getcwd()
output_path = os.path.join(project_folder, output_file_name)

# Step 4: Download the file
print(f"Downloading file from Google Drive...")
gdown.download(gdrive_url, output_path, quiet=False)
print(f"File downloaded and saved as: {output_path}")

# Step 5: Load the CSV into a DataFrame (optional)
gdp_df = pd.read_csv(output_path)
print("Loaded DataFrame:")
print(gdp_df.head())

print(type(gdp_df))

import pandas as pd

## Assigning the column Date as index ##

# Ensure the "Date" column exists before proceeding
if 'Date' in gdp_df.columns:
    try:
        # Convert 'Date' to datetime format if not already in datetime64[ns]
        if not pd.api.types.is_datetime64_any_dtype(gdp_df['Date']):
            gdp_df['Date'] = pd.to_datetime(gdp_df['Date'], errors='coerce')

        # Drop rows with invalid or missing dates after conversion
        if gdp_df['Date'].isna().any():
            print("Warning: Some dates could not be converted and will be dropped.")
            gdp_df = gdp_df.dropna(subset=['Date'])

        # Set 'Date' as the index only if it's not already set
        if gdp_df.index.name != 'Date':
            gdp_df = gdp_df.set_index('Date')
            print("Successfully set 'Date' as the index.")
        else:
            print("'Date' is already the index.")

    except Exception as e:
        print(f"Error during Date processing: {e}")
else:
    print("Error: The 'Date' column does not exist in the DataFrame.")



print(gdp_df.index)

print(gdp_df)

## Conversion of pandas.DataFrame into a pandas.Series ##
###########################################################


# # Optional: Manipulate the data or save it again (if needed)
# processed_file_path = os.path.join(project_folder, 'Processed_GDP_Percent_Change.csv')
# gdp_df.to_csv(processed_file_path, index=False)
# print(f"Processed file saved at: {processed_file_path}")


##

# Example of loading the data (replace this with your actual data source)
# gdp_growth = pd.Series([...], index=pd.period_range('1962Q1', '2024Q2', freq='Q'))
# quarterly_factors = pd.DataFrame([...], index=pd.period_range('1962Q1', '2024Q2', freq='Q'))

#### Linear model (pandas.series as input for gdp quarterly growth rate) ####
# Historical cutoff date for the initial estimation
# initial_cutoff_date = '1999-Q4'
#
# # Define the last forecast month
# last_forecast_month = '2024-07'
#
# # Generate all vintage months starting from January 2000 up to July 2024
# vintages = pd.date_range(start='2000-01', end=last_forecast_month, freq='M')
#
# # Initialize a dictionary to store nowcasts for each quarter
# nowcasts = {}
#
# # Iterate over each "vintage" month
# for vintage in vintages:
#     # Print current vintage being processed
#     print(f"\nProcessing vintage: {vintage.strftime('%Y-%m')}")
#
#     # Determine the cutoff for historical estimation: last quarter before this vintage
#     cutoff_quarter = pd.Period(vintage, freq='Q') - 1
#
#     # Step 1: Estimate the OLS model using data up to the cutoff quarter
#     historical_data = gdp_growth.loc[:cutoff_quarter]
#     historical_factors = quarterly_factors.loc[:cutoff_quarter]
#
#     # Ensure factors are aligned with GDP growth
#     X_train = historical_factors
#     y_train = historical_data
#
#     # Add a constant for the intercept
#     X_train = sm.add_constant(X_train)
#
#     # Estimate the OLS model
#     model = sm.OLS(y_train, X_train).fit()
#
#     # Step 2: Define the next two quarters to forecast
#     forecast_quarters = [cutoff_quarter + 1, cutoff_quarter + 2]
#
#     for forecast_quarter in forecast_quarters:
#         # Check if latent factors for the forecast quarter are available
#         if forecast_quarter in quarterly_factors.index:
#             # Prepare the input data for prediction
#             X_forecast = quarterly_factors.loc[forecast_quarter]
#             X_forecast = sm.add_constant(X_forecast)
#
#             # Predict GDP growth for the forecast quarter
#             nowcast = model.predict(X_forecast)[0]
#
#             # Print the nowcast result
#             print(f"Nowcast for {forecast_quarter}: {nowcast:.4f}")
#
#             # Store the nowcast in the results dictionary
#             if forecast_quarter not in nowcasts:
#                 nowcasts[forecast_quarter] = []
#             nowcasts[forecast_quarter].append({
#                 'vintage': vintage.strftime('%Y-%m'),
#                 'nowcast': nowcast
#             })
#
# # Output all nowcasts
# print("\nNowcasts Summary:")
# for quarter, forecasts in nowcasts.items():
#     print(f"\nQuarter: {quarter}")
#     for forecast in forecasts:
#         print(f"Vintage: {forecast['vintage']}, Nowcast: {forecast['nowcast']:.4f}")

### Method adapted to accept a dataframe instead of a series ##

import pandas as pd
import statsmodels.api as sm

# Historical cutoff date for the initial estimation
initial_cutoff_date = '1999-Q4'

# Define the last forecast month
last_forecast_month = '2024-07'

# Generate all vintage months starting from January 2000 up to July 2024
vintages = pd.date_range(start='2000-01', end=last_forecast_month, freq='M')

# Initialize a dictionary to store nowcasts for each quarter
nowcasts = {}

# Specify the target column name for GDP growth rates
gdp_growth_column = 'growth_rate'

# Iterate over each "vintage" month
for vintage in vintages:
    # Print current vintage being processed
    print(f"\nProcessing vintage: {vintage.strftime('%Y-%m')}")

    # Determine the cutoff for historical estimation: last quarter before this vintage
    cutoff_quarter = pd.Period(vintage, freq='Q') - 1

    # Step 1: Estimate the OLS model using data up to the cutoff quarter
    # Extract GDP growth as a Series using the specified column name
    historical_data = gdp_growth.loc[:cutoff_quarter, gdp_growth_column]
    historical_factors = quarterly_factors.loc[:cutoff_quarter]

    # Ensure factors are aligned with GDP growth
    X_train = historical_factors
    y_train = historical_data

    # Add a constant for the intercept
    X_train = sm.add_constant(X_train)

    # Estimate the OLS model
    model = sm.OLS(y_train, X_train).fit()

    # Step 2: Define the next two quarters to forecast
    forecast_quarters = [cutoff_quarter + 1, cutoff_quarter + 2]

    for forecast_quarter in forecast_quarters:
        # Check if latent factors for the forecast quarter are available
        if forecast_quarter in quarterly_factors.index:
            # Prepare the input data for prediction
            X_forecast = quarterly_factors.loc[forecast_quarter]
            X_forecast = sm.add_constant(X_forecast)

            # Predict GDP growth for the forecast quarter
            nowcast = model.predict(X_forecast)[0]

            # Print the nowcast result
            print(f"Nowcast for {forecast_quarter}: {nowcast:.4f}")

            # Store the nowcast in the results dictionary
            if forecast_quarter not in nowcasts:
                nowcasts[forecast_quarter] = []
            nowcasts[forecast_quarter].append({
                'vintage': vintage.strftime('%Y-%m'),
                'nowcast': nowcast
            })

# Output all nowcasts
print("\nNowcasts Summary:")
for quarter, forecasts in nowcasts.items():
    print(f"\nQuarter: {quarter}")
    for forecast in forecasts:
        print(f"Vintage: {forecast['vintage']}, Nowcast: {forecast['nowcast']:.4f}")
