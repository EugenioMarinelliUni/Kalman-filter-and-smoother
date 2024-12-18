#!/usr/bin/env python
# coding: utf-8

# # Libraries and dataset

# ## Importing the libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().run_line_magic('pip', 'install google-colab')



# In[ ]:


# general libraries
import io
from google.colab import files # (used to import dataset)
import numpy as np
import pandas as pd
import seaborn as sns
import csv
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import math


# In[ ]:


# time-series-specific libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
# from tsmoothie.smoother import *
# from tsmoothie.utils_func import create_windows


# In[ ]:


np.random.seed(0)


# ## Uploading files

# In[ ]:


# import dataset
import os
import requests
import pandas as pd

# Check if the dataset already exists
file_path = 'fred_md.csv'

if not os.path.exists(file_path):
    url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
    response = requests.get(url)

    with open(file_path, 'wb') as file:
        file.write(response.content)
    print("FRED-MD dataset downloaded successfully.")
else:
    print("FRED-MD dataset already exists locally.")

# Load the dataset into a DataFrame
fred_md_df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# Display the first few rows
print(fred_md_df.head())

if 'sasdate' in fred_md_df.columns:
    print("The column 'sasdate' is present in the DataFrame.")
else:
    print("The column 'sasdate' is not present in the DataFrame.")

for idx, col in enumerate(fred_md_df.columns):
    print(f"{idx}: {col}")

print(fred_md_df.index)




# From dataset to dataframe
# 

# In[ ]:


import os
import pandas as pd
get_ipython().system('pip install fredapi')
from fredapi import Fred

# Ensure required libraries are installed
# !pip install pandas fredapi

# Replace 'your_api_key_here' with your actual FRED API key
fred = Fred(api_key='feabb7180fc1f516e63d0b320e07e6dd')

# File path for the locally saved data
file_path = 'GDP_Percent_Change.csv'

# Check if the CSV file already exists
if not os.path.exists(file_path):
    # Download GDP Percent Change from FRED
    gdp_data = fred.get_series('A191RL1Q225SBEA', observation_start='1947-04-01', observation_end='2024-09-30')

    # Convert to DataFrame for easy manipulation
    gdp_df = pd.DataFrame({'Date': gdp_data.index, 'GDP Percent Change': gdp_data.values})

    # Save to CSV
    gdp_df.to_csv(file_path, index=False)
    print("Data downloaded and saved to CSV.")
else:
    # Load the data from the existing CSV file
    gdp_df = pd.read_csv(file_path)
    print("Data loaded from existing CSV file.")

# Display the first few rows
print(gdp_df.head())

for idx, col in enumerate(gdp_df.columns):
    print(f"{idx}: {col}")



# ## Data manipulation

# ### Gdp dataframe manipulation

# In[ ]:


import pandas as pd
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Check if the 'Date' column exists and handle accordingly
if 'Date' in gdp_df.columns:
    # Convert 'Date' column to datetime if not already done
    if gdp_df['Date'].dtype != 'datetime64[ns]':
        gdp_df['Date'] = pd.to_datetime(gdp_df['Date'])
        print("Date column converted to datetime format.")
    else:
        print("Date column already in datetime format.")

    # Apply the date offset transformation
    gdp_df['Date'] = gdp_df['Date'] - DateOffset(months=1)
    gdp_df = gdp_df.set_index('Date')
    print("Date offset applied, and 'Date' set as index.")
else:
    print("Error: 'Date' column is not present in the DataFrame.")

for col in gdp_df.columns:
    print(col)

## Checking for missing values in the GDP series ##
column_names = gdp_df.columns
print(column_names)
missing_data = gdp_df[gdp_df['GDP Percent Change'].isnull()]
indexes = missing_data.index.tolist()
print(indexes)

# Function to check stationarity and transform if necessary
def ensure_stationarity(series, name):
    adf_test = adfuller(series.dropna())
    print(f"{name}: ADF Statistic={adf_test[0]}, p-value={adf_test[1]}")

    if adf_test[1] < 0.05:
        print(f"{name} is stationary.")
        return series  # Already stationary
    else:
        print(f"{name} is not stationary. Differencing will be applied.")
        return series.diff().dropna()  # Apply differencing to ensure stationarity

# Ensure GDP stationarity
gdp_df['GDP Percent Change'] = ensure_stationarity(gdp_df['GDP Percent Change'], 'GDP Percent Change')

# Display the first few rows
print(gdp_df.head())


# ### Fred Md dataframe manipulations

# In[ ]:


import os
import pandas as pd
from fredapi import Fred

# Initialize FRED API
fred = Fred(api_key='feabb7180fc1f516e63d0b320e07e6dd')

# Display column indexes for verification
for idx, col in enumerate(fred_md_df.columns):
    print(f"{idx}: {col}")



# Check if the file 'modified_fred_md_df.csv' already exists
if os.path.exists('modified_fred_md_df.csv'):
    # Load the existing file
    modified_fred_md_df = pd.read_csv('modified_fred_md_df.csv', index_col='Date', parse_dates=True)
    print("Dataframe ready. File with 'Date' as index already exists.")
else:
    try:
        # Check if 'sasdate' is the index
        if fred_md_df.index.name == 'sasdate':
            # Ensure 'sasdate' becomes a column
            fred_md_df.reset_index(inplace=True)
            print("'sasdate' was an index and has been reset to a column.")

            # Perform initial transformations
            fred_md_df.drop(fred_md_df.index[:1], inplace=True)
            fred_md_df.rename(columns={'sasdate': 'Date'}, inplace=True)
            fred_md_df['Date'] = pd.to_datetime(fred_md_df['Date'])
            fred_md_df.set_index('Date', inplace=True)
            fred_md_df.sort_index(inplace=True)
            print("Initial transformations completed.")

            # Additional operations: Remove specified series and add new series
            indexes_to_remove = [19, 20, 57, 93, 121]
            fred_md_df.drop(fred_md_df.columns[indexes_to_remove], axis=1, inplace=True)
            print("Specified series removed by index.")

            # Save the modified DataFrame
            fred_md_df.to_csv('modified_fred_md_df.csv')
            print("Dataset updated and saved as 'modified_fred_md_df.csv'.")
            modified_fred_md_df = fred_md_df
        else:
            print("Error: 'sasdate' index not found in the DataFrame.")

    except KeyError as e:
        print(f"KeyError: {e} - The specified columns or indexes might be missing.")
    except ValueError as e:
        print(f"ValueError: {e} - Issue with data type conversion or merging.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Display the columns of the modified DataFrame to confirm successful transformation
print("\nColumns in the modified DataFrame:")
for idx, col in enumerate(modified_fred_md_df.columns):
     print(f"{idx}: {col}")

# # Display the first few rows of the modified dataset
# print(fred_md_df.head())

# # Display column indexes for verification
# for idx, col in enumerate(fred_md_df.columns):
#     print(f"{idx}: {col}")

# # Display column indexes for verification
#



# ### Transforming the series to ensure stationarity; grouping the modified  series by type (Soybilgen and Yazgan, 2021)
# 
# ---
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load your DataFrame
df = pd.read_csv('modified_fred_md_df.csv', index_col=0, parse_dates=True)
print(df.head())
# # Filter data to include only from January 2000 onward
# df = df.loc['2000-01-01':]

##############################################################################

## Step 1: Removing NONBORRES, adding BORROW, then performing checks ##
## Why did I decide to remove NONBORRES? Due to the presence of negative values, that made impossible to apply the proposed transformation (monthly growth rate).
## I decided to drop this series and replace it with BORROW, due to the relation Nonborrowed reserves (NONBORRES) equals total reserves (TOTRESNS), less total borrowings from the Federal Reserve (BORROW).

# Replace NONBORRES with BORROW
nonborres_index = df.columns.get_loc("NONBORRES")  # Get the index of NONBORRES
nonborres_column = df.columns[nonborres_index]    # Get the column name

print(f"Index of NONBORRES: {nonborres_index}")
print(f"Column name of NONBORRES: {nonborres_column}")

##### Insert BORROW as new column####

import pandas as pd
from pandas_datareader import data as pdr

import pandas as pd
from fredapi import Fred

### Step 1: Fetch the entire BORROW series from FRED using fredapi ###

fred = Fred(api_key="feabb7180fc1f516e63d0b320e07e6dd")
borrow_series = fred.get_series("BORROW")

### Step 2: Inspect the BORROW series ###
print("First few rows of BORROW series:")
print(borrow_series.head())

print("\nIndex of BORROW series:")
print(borrow_series.index)

print("\nSummary of BORROW series:")
print(borrow_series.info())

### Step 3: Align BORROW series with the existing DataFrame's index ###
### Filter BORROW series to start from the earliest date in df ###
oldest_date_in_df = df.index.min()
print(f"Earliest date in df: {oldest_date_in_df}")
borrow_series_filtered = borrow_series.loc[oldest_date_in_df:]
print(f"Filtered BORROW series (from {oldest_date_in_df}): {borrow_series_filtered}")

### Step 4: Reindex the BORROW series to match the index of df ###
borrow_series_filtered = borrow_series_filtered.reindex(df.index)

### Step 5: Add the BORROW series as a new column in your DataFrame ###
df['BORROW'] = borrow_series_filtered

### Step 6: Check the columns of the DataFrame ###
print("\nColumns in the DataFrame:")
print(df.columns)

print("\nColumns in the modified DataFrame:")
for idx, col in enumerate(df.columns):
     print(f"{idx}: {col}")

### Step 7: Print the last column of the DataFrame (which corresponds to the last series added) ###
print("\nLast column (series added):")
print(df.iloc[:, -1])  # Access the last column using iloc

### Step 8: Print a list of column names ###
print("\nList of column names:")
print(list(df.columns))

### Drop NONBORRES ###
df.drop(columns=[nonborres_column], inplace=True)

### Step 8: Print a list of column names ###
print("\nList of column names:")
print(list(df.columns))
print(df)
################################################################################

## Step 2: Define the indices for each transformation ##
monthly_growth_rate_indices = list(range(0, 18)) + [19, 20] + list(range(23, 42)) + list(range(45, 59)) + list(range(60, 72)) + list(range(89, 93)) + list(range(116, 121))
monthly_difference_indices = [18, 21, 22] + list(range(42, 45)) + [59] + list(range(72, 81))
monthly_diff_yearly_growth_rate_indices = list(range(93, 116))
no_transformation_indices = list(range(81, 89))
print(monthly_growth_rate_indices)
print(monthly_difference_indices)
print(monthly_diff_yearly_growth_rate_indices)
print(no_transformation_indices)
################################################################################

## Step 3: Checking for the presence of missing values and non-positive values in the dataframe ##
import pandas as pd
import numpy as np

# Create a report for missing and non-positive values
report = {}

# Check for missing and non-positive values
for group_name, indices in {
    "monthly_growth_rate": monthly_growth_rate_indices,
    "monthly_difference": monthly_difference_indices,
    "monthly_diff_yearly_growth_rate": monthly_diff_yearly_growth_rate_indices,
    "no_transformation": no_transformation_indices,
}.items():
    group_report = {}
    for idx in indices:
        column_name = df.columns[idx]
        column_data = df.iloc[:, idx]

        # Missing values
        missing = column_data.isna()
        missing_periods = column_data[missing].index.tolist()

        # Non-positive values (only for log-transformed columns)
        if group_name in {"monthly_growth_rate", "monthly_diff_yearly_growth_rate"}:
            non_positive = column_data <= 0
            non_positive_periods = column_data[non_positive].index.tolist()
        else:
            non_positive_periods = []

        # Record issues
        group_report[column_name] = {
            "missing_count": missing.sum(),
            "missing_periods": missing_periods,
            "non_positive_count": len(non_positive_periods),
            "non_positive_periods": non_positive_periods,
        }

    report[group_name] = group_report

# Display the report
for group_name, group_report in report.items():
    print(f"Group: {group_name}")
    for column_name, column_report in group_report.items():
        print(f"  Column: {column_name}")
        print(f"    Missing values: {column_report['missing_count']}")
        if column_report['missing_periods']:
            print(f"    Missing periods: {column_report['missing_periods']}")
        if "non_positive_count" in column_report:
            print(f"    Non-positive values: {column_report['non_positive_count']}")
            if column_report['non_positive_periods']:
                print(f"    Non-positive periods: {column_report['non_positive_periods']}")
        print()

#################### EXECUTE CODE SEQUENTIALLY UNTIL THIS LINE #########################################

import pandas as pd

def interpolate_and_save(df, column_name, additional_column, missing_date, output_file):
    """
    Fills missing values in specified columns using linear interpolation and saves the updated DataFrame to a file.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the primary column to interpolate.
        additional_column (str): The name of the additional column to interpolate.
        missing_date (str): The date of the missing value in 'YYYY-MM-DD' format.
        output_file (str): The file path to save the updated DataFrame.
    """
    # Ensure the index is a datetime type for interpolation
    df.index = pd.to_datetime(df.index)

    # Perform linear interpolation on the specified columns
    df[column_name] = df[column_name].interpolate(method='linear')
    df[additional_column] = df[additional_column].interpolate(method='linear')

    # Verify the missing values are filled
    interpolated_value_main = df.loc[missing_date, column_name]
    interpolated_value_additional = df.loc[missing_date, additional_column]

    print(f"Interpolated value for {column_name} on {missing_date}: {interpolated_value_main}")
    print(f"Interpolated value for {additional_column} on {missing_date}: {interpolated_value_additional}")

    # Save the updated DataFrame to a file
    df.to_csv(output_file)
    print(f"Updated DataFrame saved to {output_file}")

# Example usage
interpolate_and_save(
    df,  # The existing DataFrame
    column_name="CP3Mx",  # Primary column to interpolate
    additional_column="COMPAPFFx",  # Additional column to interpolate
    missing_date="2020-04-01",  # Date of the missing value
    output_file="updated_dataframe.csv"  # File to save the updated DataFrame
)


#######################

import pandas as pd

def check_missing_values_from_csv(file_path):
    """
    Loads a DataFrame from a CSV file and checks for missing values,
    specifying the column indexes and dates with missing data.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        None: Prints the missing data information.
    """
    # Load the DataFrame from the CSV file
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    print(f"Loaded DataFrame from {file_path} with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Check for missing values
    missing_data = df.isna()

    # Iterate through columns and find missing values
    for idx, column in enumerate(df.columns):
        missing_dates = missing_data.index[missing_data[column]].tolist()
        if missing_dates:
            print(f"Column Index: {idx}, Column Name: {column}")
            print(f"  Missing Dates: {missing_dates}")

# Example Usage
csv_file_path = 'updated_dataframe.csv'  # Replace with the actual path to your CSV file
check_missing_values_from_csv(csv_file_path)









# In[ ]:


#### Optional steps to be performed in case the series require to be seasonally adjusted (part 1) ####

############ Utility method for webscraping #########################

import requests
from bs4 import BeautifulSoup

def check_series_seasonality_and_frequency(series_name):
    """
    Check if the given FRED series is 'Not Seasonally Adjusted' or 'Seasonally Adjusted'
    and determine its frequency based on specific DOM elements.
    Args:
        series_name (str): The name of the FRED series.
    Returns:
        dict: A dictionary with the seasonality and frequency information.
    """
    # Construct the URL for the FRED series page
    url = f"https://fred.stlouisfed.org/series/{series_name}"

    # Fetch the web page
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data for {series_name}: HTTP {response.status_code}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Selector for seasonality (corresponding to XPath /html/body/div/div[1]/div/div[3]/div[1]/div[3]/span)
    seasonality_section = soup.select_one('html > body > div > div:nth-of-type(1) > div > div:nth-of-type(3) > div:nth-of-type(1) > div:nth-of-type(3) > span')
    if not seasonality_section:
        print(f"Seasonality section not found for {series_name}")
        return None

    # Selector for frequency (corresponding to XPath /html/body/div/div[1]/div/div[3]/div[1]/div[4]/span/span)
    frequency_section = soup.select_one('html > body > div > div:nth-of-type(1) > div > div:nth-of-type(3) > div:nth-of-type(1) > div:nth-of-type(4) > span > span')
    if not frequency_section:
        print(f"Frequency section not found for {series_name}")
        return None

    # Extract text for seasonality and frequency
    seasonality_text = seasonality_section.get_text(strip=True)
    frequency_text = frequency_section.get_text(strip=True)

    # Determine seasonality
    if "Not Seasonally Adjusted" in seasonality_text:
        seasonality = "Not Seasonally Adjusted"
    elif "Seasonally Adjusted Annual Rate" in seasonality_text or "Seasonally Adjusted" in seasonality_text:
        seasonality = "Seasonally Adjusted"
    else:
        seasonality = "Unknown"

    # Determine frequency
    if "Monthly" in frequency_text:
        frequency = "Monthly"
    elif "Quarterly" in frequency_text:
        frequency = "Quarterly"
    elif "Annual" in frequency_text:
        frequency = "Annual"
    else:
        frequency = "Other"

    return {"series_name": series_name, "seasonality": seasonality, "frequency": frequency}

# Example DataFrame (replace with your actual DataFrame)
df_columns = df.columns.tolist()  # Replace df with your actual DataFrame variable

# Create a report for each series in the DataFrame
report = []
for series in df_columns:
    print(f"Checking series: {series}")
    result = check_series_seasonality_and_frequency(series)
    if result:
        report.append(result)

# Display the report
for entry in report:
    print(f"Series: {entry['series_name']}, Seasonality: {entry['seasonality']}, Frequency: {entry['frequency']}")



# In[ ]:


#### Optional steps to be performed in case the series require to be seasonally adjusted (part 2) ####

# Manually categorize series that were not found
not_found_seasonally_adjusted = [
    "CMRMTSPLx", "RETAILx", "CLAIMSx", "AMDMNOx", "ANDENOx",
    "AMDMUOx", "BUSINVx", "ISRATIOx", "CONSPI"
]

not_found_not_seasonally_adjusted = [
    "S&P 500", "S&P div yield", "S&P PE ratio", "CP3Mx", "COMPAPFFx",
    "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "OILPRICEx", "VIXCLSx"
]

# Separate found series based on seasonality
seasonally_adjusted = []
not_seasonally_adjusted = []

for entry in report:
    if entry['seasonality'] == "Seasonally Adjusted":
        seasonally_adjusted.append(entry['series_name'])
    elif entry['seasonality'] == "Not Seasonally Adjusted":
        not_seasonally_adjusted.append(entry['series_name'])

# Add manually specified series to the appropriate groups
seasonally_adjusted.extend(not_found_seasonally_adjusted)
not_seasonally_adjusted.extend(not_found_not_seasonally_adjusted)

# Display results
print("Seasonally Adjusted Series:")
print(seasonally_adjusted)

print("\nNot Seasonally Adjusted Series:")
print(not_seasonally_adjusted)
##########################################
######## Validation method ###############
##########################################
# Combine grouped lists for validation
grouped_list = seasonally_adjusted + not_seasonally_adjusted

# Get the list of column names from the DataFrame
df_columns = list(df.columns)

# Check for any column names not included in the grouped lists
missing_columns = [col for col in df_columns if col not in grouped_list]

# Report the results
if missing_columns:
    print("The following column names are missing from both arrays:")
    print(missing_columns)
else:
    print("All column names are included in one of the arrays.")



# ### Appplying the transformations to the series following the methodology used in the paper.

# In[ ]:


#####################################################################
############ RESTART EXECUTING CODE FROM HERE! ######################
#####################################################################

######################################################################################################

# Apply Log-Transformed Monthly Growth Rate transformation (perform after adjusting for seasonality) TRANSFORMATION 1

######################################################################################################

df = pd.read_csv('updated_dataframe.csv', index_col='Date', parse_dates=True)

# Print basic information about the loaded DataFrame
print(f"Loaded DataFrame type: {type(df)}")
print(f"Loaded DataFrame shape: {df.shape}")

import pandas as pd

# Dictionary to store transformed columns
monthly_growth_rate_columns = {}

# Perform the monthly growth rate calculation
for idx in monthly_growth_rate_indices:
    # Get the column name
    column_name = df.columns[idx]

    # Calculate the growth rate based on adjacent rows (dates)
    growth_rate = (df.iloc[:, idx] - df.iloc[:, idx].shift(1)) / df.iloc[:, idx].shift(1)

    # Add the transformed column to the dictionary for creating a new DataFrame
    monthly_growth_rate_columns[column_name] = growth_rate

# Create a new DataFrame with only the transformed series
monthly_growth_rate_df = pd.DataFrame(monthly_growth_rate_columns, index=df.index)  # Retain the original time index

# Save the new DataFrame to a CSV file
monthly_growth_rate_df.to_csv("monthly_growth_rate_transformed.csv", index=True)  # Save with the index

# Print the transformed DataFrame
print("Transformed DataFrame (Monthly Growth Rates):")
print(monthly_growth_rate_df)


########### Checking dates with missing values #########

def extract_missing_dates_and_columns(monthly_growth_rate_df):
    # Dictionary to hold missing columns grouped by missing dates
    missing_dates_columns = {}

    # Iterate over each row (date) to find missing values
    for date, row in monthly_growth_rate_df.iterrows():
        # Find the columns with missing values for the current date
        missing_columns = row[row.isna()].index.tolist()

        # If there are missing values, store the corresponding columns in the dictionary
        if missing_columns:
            missing_dates_columns[date] = missing_columns

    return missing_dates_columns

# Example usage
missing_dates_columns = extract_missing_dates_and_columns(monthly_growth_rate_df)

# Print the missing dates and corresponding columns
if missing_dates_columns:
    print("Missing dates and corresponding columns:")
    for date, columns in missing_dates_columns.items():
        print(f"Date: {date.strftime('%Y-%m-%d')} -> Missing columns: {columns}")
else:
    print("No missing data found for any dates.")

### The diagnostic tool reveals the presence of three columns with a significant amount of missing data: 'AMDMNOx', 'ANDENOx', 'AMDMUOx'

######################################################################################################

# Apply Monthly difference transformation (perform after adjusting for seasonality) TRANSFORMATION 2

######################################################################################################

monthly_diff_columns = {}

# Apply Monthly Difference transformation (without modifying the original DataFrame)
for idx in monthly_difference_indices:
    # Get the column name
    column_name = df.columns[idx]

    # Calculate the monthly difference
    monthly_difference = df.iloc[:, idx].diff()

    # Add the transformed column to the dictionary
    monthly_diff_columns[column_name] = monthly_difference

# Create a new DataFrame with only the affected columns (monthly differences)
monthly_diff_df = pd.DataFrame(monthly_diff_columns, index=df.index)  # Retain the original time index

# Save the new DataFrame to a CSV file
monthly_diff_df.to_csv("monthly_diff_transformed.csv", index=True)  # Save with the index

# Print the transformed DataFrame (monthly differences)
print("Transformed DataFrame (Monthly Differences):")
print(monthly_diff_df)


####

def extract_missing_dates_and_columns(monthly_diff_df):
    # Dictionary to hold missing columns grouped by missing dates
    missing_dates_columns = {}

    # Iterate over each row (date) to find missing values
    for date, row in monthly_diff_df.iterrows():
        # Find the columns with missing values for the current date
        missing_columns = row[row.isna()].index.tolist()

        # If there are missing values, store the corresponding columns in the dictionary
        if missing_columns:
            missing_dates_columns[date] = missing_columns

    return missing_dates_columns

# Example usage
missing_dates_columns = extract_missing_dates_and_columns(monthly_diff_df)

# Print the missing dates and corresponding columns
if missing_dates_columns:
    print("Missing dates and corresponding columns:")
    for date, columns in missing_dates_columns.items():
        print(f"Date: {date.strftime('%Y-%m-%d')} -> Missing columns: {columns}")
else:
    print("No missing data found for any dates.")


######################################################################################################

# Apply Monthly differences of yearly growth rate transformation (perform after adjusting for seasonality) TRANSFORMATION 3

######################################################################################################

import numpy as np
import pandas as pd

# Initialize a dictionary to store transformed columns
transformed_monthly_diff_growth_rate_columns = {}

# Apply Monthly Difference of Yearly Growth Rate transformation (log-transformed)
for idx in monthly_diff_yearly_growth_rate_indices:
    # Get the column name
    column_name = df.columns[idx]

    # Log-transform the series to stabilize variance
    log_transformed_series = np.log(df.iloc[:, idx])

    # Calculate the yearly growth rate: Difference between log-transformed values lagged by 12 months
    yearly_growth_rate = log_transformed_series - log_transformed_series.shift(12)  # Shift by 12 months for yearly growth rate

    # Calculate the monthly difference of the yearly growth rate
    monthly_diff_of_yearly_growth_rate = yearly_growth_rate - yearly_growth_rate.shift(1)  # Shift by 1 month for monthly difference

    # Add the transformed column to the dictionary (without modifying the original DataFrame)
    transformed_monthly_diff_growth_rate_columns[column_name] = monthly_diff_of_yearly_growth_rate

    # Optionally, you can print the transformed column to check
    print(f"Transformed column for {column_name}:")
    print(monthly_diff_of_yearly_growth_rate.head())  # Print the first few rows of the transformed series

# Create a new DataFrame for the transformed columns (monthly difference of yearly growth rates)
transformed_monthly_diff_growth_rate_df = pd.DataFrame(
    transformed_monthly_diff_growth_rate_columns,
    index=df.index  # Retain the original time index
)

# Save the new DataFrame to a CSV file
transformed_monthly_diff_growth_rate_df.to_csv("monthly_diff_yearly_growth_rate_transformed.csv", index=True)  # Save with the index

# Print the transformed DataFrame
print("Transformed DataFrame (Monthly Difference of Yearly Growth Rate):")
print(transformed_monthly_diff_growth_rate_df)

#####

def extract_missing_dates_and_columns(transformed_monthly_diff_growth_rate_df):
    # Dictionary to hold missing columns grouped by missing dates
    missing_dates_columns = {}

    # Iterate over each row (date) to find missing values
    for date, row in transformed_monthly_diff_growth_rate_df.iterrows():
        # Find the columns with missing values for the current date
        missing_columns = row[row.isna()].index.tolist()

        # If there are missing values, store the corresponding columns in the dictionary
        if missing_columns:
            missing_dates_columns[date] = missing_columns

    return missing_dates_columns

# Example usage
missing_dates_columns = extract_missing_dates_and_columns(transformed_monthly_diff_growth_rate_df)

# Print the missing dates and corresponding columns
if missing_dates_columns:
    print("Missing dates and corresponding columns:")
    for date, columns in missing_dates_columns.items():
        print(f"Date: {date.strftime('%Y-%m-%d')} -> Missing columns: {columns}")
else:
    print("No missing data found for any dates.")


################################################################################

import pandas as pd

# Method to print the entire column associated with a specific index in the transformed DataFrame
def print_transformed_column_full(df, column_index):
    """
    Print the entire column associated with the given column index from the transformed DataFrame
    without truncating the output.

    Args:
        df (pd.DataFrame): The DataFrame containing transformed data.
        column_index (str): The column index (name) to retrieve and print.
    """
    if column_index in df.columns:
        # Temporarily change Pandas display settings to show all rows
        with pd.option_context('display.max_rows', None):  # Set 'max_rows' to None to display all rows
            print(f"Column '{column_index}' in transformed data:")
            print(df[column_index])  # Print the entire column without truncation
    else:
        print(f"Column '{column_index}' not found in the DataFrame.")

# Example usage: Print the entire column for index 'WPSFD49207'
column_index = 'WPSFD49207'
print_transformed_column_full(transformed_monthly_diff_growth_rate_df, column_index)

#####################################################################################
### Putting together the transformed dateframes preserving the correct time index ###
#####################################################################################

import pandas as pd

def read_original_data(original_file):
    """Read the original DataFrame to preserve its index order."""
    print("Step 1: Reading the original DataFrame...")
    original_df = pd.read_csv(original_file, index_col=0)  # Ensure the original index is used
    print("Original DataFrame (first 5 rows):")
    print(original_df.head())  # Print the first few rows for inspection
    return original_df

def read_transformed_data(growth_rate_file, diff_file, yearly_growth_rate_file):
    """Read the transformed DataFrames."""
    print("\nStep 2: Reading the transformed DataFrames...")
    monthly_growth_rate_df = pd.read_csv(growth_rate_file, index_col=0)
    monthly_diff_df = pd.read_csv(diff_file, index_col=0)
    monthly_diff_yearly_growth_rate_df = pd.read_csv(yearly_growth_rate_file, index_col=0)

    print("\nMonthly Growth Rate DataFrame (first 5 rows):")
    print(monthly_growth_rate_df.head())  # Print the first few rows for inspection

    print("\nMonthly Difference DataFrame (first 5 rows):")
    print(monthly_diff_df.head())  # Print the first few rows for inspection

    print("\nMonthly Difference of Yearly Growth Rate DataFrame (first 5 rows):")
    print(monthly_diff_yearly_growth_rate_df.head())  # Print the first few rows for inspection

    return monthly_growth_rate_df, monthly_diff_df, monthly_diff_yearly_growth_rate_df

def identify_columns(original_df, monthly_growth_rate_df, monthly_diff_df, monthly_diff_yearly_growth_rate_df):
    """Identify untransformed columns from the original DataFrame."""
    print("\nStep 3: Identifying untransformed columns...")
    transformed_columns = (
        set(monthly_growth_rate_df.columns) |
        set(monthly_diff_df.columns) |
        set(monthly_diff_yearly_growth_rate_df.columns)
    )
    untransformed_columns = [col for col in original_df.columns if col not in transformed_columns]

    print("\nTransformed Columns:")
    print(transformed_columns)  # Print the set of transformed columns

    print("\nUntransformed Columns:")
    print(untransformed_columns)  # Print the list of untransformed columns

    return untransformed_columns

def extract_untransformed_columns(original_df, untransformed_columns):
    """Extract the untransformed columns from the original DataFrame."""
    print("\nStep 4: Extracting untransformed columns...")
    untransformed_df = original_df[untransformed_columns]
    print("\nUntransformed DataFrame (first 5 rows):")
    print(untransformed_df.head())  # Print the untransformed columns for inspection
    return untransformed_df

# Example usage:
# Call each method step by step to visualize the intermediate results
original_df = read_original_data("updated_dataframe.csv")
monthly_growth_rate_df, monthly_diff_df, monthly_diff_yearly_growth_rate_df = read_transformed_data(
    "monthly_growth_rate_transformed.csv", "monthly_diff_transformed.csv", "monthly_diff_yearly_growth_rate_transformed.csv"
)
untransformed_columns = identify_columns(
    original_df, monthly_growth_rate_df, monthly_diff_df, monthly_diff_yearly_growth_rate_df
)
untransformed_df = extract_untransformed_columns(original_df, untransformed_columns)

############

import pandas as pd

def extract_and_associate_indices(original_file):
    """Extract indices and associate them with the column names."""
    print("Step: Extracting and associating indices with column names...")

    # Read the original DataFrame
    original_df = pd.read_csv(original_file, index_col=0)  # Ensure the original index is used

    # Extract the index (row labels) and associate them with column names
    index_column_mapping = {idx: col for idx, col in enumerate(original_df.columns)}

    # Print the index-column association
    print("\nIndex to Column Mapping:")
    print(index_column_mapping)

    return index_column_mapping

# Example usage:
index_column_mapping = extract_and_associate_indices("updated_dataframe.csv")

##################

import pandas as pd

def combine_dataframes(untransformed_df, monthly_growth_rate_df, monthly_diff_df, monthly_diff_yearly_growth_rate_df):
    """Combine the untransformed and transformed DataFrames, preserving the time index."""
    print("Step: Combining the DataFrames while preserving the time index...")

    # Combine all DataFrames along the columns (axis=1)
    combined_df = pd.concat([
        untransformed_df,
        monthly_growth_rate_df,
        monthly_diff_df,
        monthly_diff_yearly_growth_rate_df
    ], axis=1)

    # Print the combined DataFrame (first 5 rows) to verify
    print("\nCombined DataFrame with all series (first 5 rows):")
    print(combined_df.head())  # Print the first few rows for inspection

    return combined_df

# Example usage:
# Assuming you already have the DataFrames: untransformed_df, monthly_growth_rate_df, monthly_diff_df, monthly_diff_yearly_growth_rate_df
combined_df = combine_dataframes(untransformed_df, monthly_growth_rate_df, monthly_diff_df, monthly_diff_yearly_growth_rate_df)

#######

import pandas as pd

def reorder_columns_by_original_order(combined_df, original_df):
    """Reorder the columns of combined_df to match the order of columns in original_df."""
    print("Step: Reordering columns in combined_df to match the original DataFrame's column order...")

    # Extract the column order from the original DataFrame
    original_column_order = original_df.columns

    # Reorder the columns in combined_df to match the original column order
    reordered_combined_df = combined_df[original_column_order]

    # Print the reordered DataFrame (first 5 rows) to verify
    print("\nReordered Combined DataFrame (first 5 rows):")
    print(reordered_combined_df.head())

    return reordered_combined_df

# Example usage:
# Assuming you already have the DataFrames: combined_df and original_df
reordered_combined_df = reorder_columns_by_original_order(combined_df, original_df)

####

def save_reordered_combined_df(reordered_combined_df, file_name="reordered_combined_data.csv"):
    """Save the reordered combined DataFrame to a CSV file."""
    print(f"\nSaving the reordered combined DataFrame to {file_name}...")

    # Save the DataFrame to a CSV file with the index
    reordered_combined_df.to_csv(file_name, index=True)

    print(f"\nReordered DataFrame saved successfully to {file_name}.")

# Example usage:
save_reordered_combined_df(reordered_combined_df)

import pandas as pd

# Check for NaN values
nan_data = reordered_combined_df.isna()

# Loop through rows and print missing columns
for date, row in nan_data.iterrows():
    missing_columns = row[row].index.tolist()  # Get columns with NaN values
    if missing_columns:  # If any columns have NaN values
        print(f"Date: {date} | Missing Columns: {missing_columns}")


################################################################################


################################################################################


import pandas as pd

# Step 1: Drop columns with more than 50 NaN values from reordered_combined_df
columns_to_drop = reordered_combined_df.columns[reordered_combined_df.isna().sum() > 50]  # Find columns with more than 50 NaNs
columns_removed_indices = reordered_combined_df.columns.get_indexer(columns_to_drop).tolist()  # Get column indices for dropped columns

# Drop the identified columns
df_cleaned = reordered_combined_df.drop(columns=columns_to_drop)

# Save the intermediate DataFrame to a CSV file
df_cleaned.to_csv("cleaned_after_column_dropping.csv", index=True)  # Save with the index

# Print the columns removed and their indices
print(f"Columns removed due to more than 50 NaN values (column names): {columns_to_drop.tolist()}")
print(f"Indices of removed columns in reordered_combined_df: {columns_removed_indices}")
print(columns_removed_indices)

# Print the saved file's name for confirmation
print("Intermediate DataFrame saved as 'cleaned_after_column_dropping.csv'")

#####

# Steps 2 & 3: create a compact dataset dropping rows with missing values for dates at the beginning and at the end of the dataframe
## (rows with missing values located at "inner" dates were already dealt with in the previous data manipulation where we used a simple linear interpolation method)

import pandas as pd

def create_compact_dataset_with_terminal_check_and_validation(input_file, output_file):
    # Step 1: Load the DataFrame
    df_cleaned = pd.read_csv(input_file, index_col=0, parse_dates=True)

    # Step 2: Determine the earliest date with complete data
    earliest_date_all_series = df_cleaned.dropna().index.min()

    # Step 3: Determine the latest date with complete data
    latest_date_all_series = df_cleaned.dropna().index.max()

    # Step 4: Filter the DataFrame
    # Keep rows between the earliest and latest dates with complete data
    df_filtered = df_cleaned.loc[earliest_date_all_series:latest_date_all_series]

    # Step 5: Save the compact dataset
    df_filtered.to_csv(output_file)
    print(f"Compact dataset saved to: {output_file}")
    print("Resulting DataFrame preview:")
    print(df_filtered.head())
    print(df_filtered.tail())

    # Step 6: Final Check for Missing Data
    missing_report = {}
    if df_filtered.isna().any().any():  # Check if there are any missing values
        print("Warning: Missing data detected in the compact dataset.")
        for column in df_filtered.columns:
            missing_dates = df_filtered[df_filtered[column].isna()].index.tolist()
            if missing_dates:
                missing_report[column] = missing_dates
                print(f"Column '{column}' has missing data on the following dates:")
                for date in missing_dates:
                    print(f"  - {date.strftime('%Y-%m-%d')}")
    else:
        print("No missing data detected in the compact dataset.")

    return df_filtered, missing_report

# Example usage
compact_df, report = create_compact_dataset_with_terminal_check_and_validation(
    "cleaned_after_column_dropping.csv",
    "compact_dataset.csv"
)

## At the end of the procedure we should obtain a "compact" dataset without missing data,
## with series starting at 1962-08-01 and ending at 2024-07-01



# Step 7: Update the grouping indices based on the cleaned DataFrame
# After dropping columns with excessive NaNs, we need to adjust the grouping indices
# Create a list of remaining column indices in df_cleaned

import pandas as pd

# Assuming reordered_combined_df is already loaded
# Example of loading the DataFrame (if not already loaded)
# reordered_combined_df = pd.read_csv('path_to_your_dataframe.csv')

# Define the original group_indices
group_indices = {
    "Output and income": [0, 1] + list(range(5, 19)),
    "Labor market": list(range(19, 45)) + list(range(113, 116)),
    "Housing": list(range(45, 55)),
    "Consumption orders and inventories": list(range(2, 5)) + list(range(55, 60)),
    "Money and credit": list(range(60, 69)) + list(range(116, 119)) + [120],
    "Interest rate": list(range(72, 81)),
    "Prices": list(range(93, 113)),
    "Stock market": list(range(69, 72)) + [119],
    "Yield spread": list(range(81, 89)),
    "Exchange rate": list(range(89, 93))
}

# Get the column labels (names) from the DataFrame
column_labels = reordered_combined_df.columns.tolist()

# Create a dictionary to store the grouped labels
grouped_labels = {}

# Group the labels based on the provided indexes in group_indices
for group_name, indices in group_indices.items():
    # Get the column names for the current group by mapping indices to column labels
    group_labels = [column_labels[idx] for idx in indices if idx < len(column_labels)]  # Ensure valid index range
    grouped_labels[group_name] = group_labels

# Print the grouped labels for each group
for group_name, labels in grouped_labels.items():
    print(f"{group_name}: {labels}")

####

columns_removed_labels = columns_to_drop.tolist()

# Print the columns to be removed and their corresponding labels
print(f"Columns to be removed (due to more than 50 NaNs): {columns_removed_labels}")
print(f"Indices of removed columns: {columns_removed_indices}")

####

for group_name, labels in grouped_labels.items():
    # Filter out the labels that are in the columns_removed_labels list
    updated_labels = [label for label in labels if label not in columns_removed_labels]
    grouped_labels[group_name] = updated_labels

# Step 5: Print the updated grouped labels after removing the unwanted columns
for group_name, labels in grouped_labels.items():
    print(f"{group_name}: {labels}")

####

import pandas as pd

# Step 1: Load the compact dataset
df_compact = pd.read_csv("compact_dataset.csv")  # Load the transformed dataframe
column_labels_updated = df_compact.columns.tolist()  # Get the column labels from the compact dataset

# Step 2: Create a reverse mapping of column labels to indices in the compact dataset
label_to_index = {label: idx for idx, label in enumerate(column_labels_updated)}

# Step 3: Convert grouped labels into grouped indices
grouped_indices = {}

# Iterate through the grouped labels and convert each label into its corresponding index
for group_name, labels in grouped_labels.items():
    # Map each label to its index using the label_to_index mapping
    updated_indices = [label_to_index[label] for label in labels if label in label_to_index]
    grouped_indices[group_name] = updated_indices

# Step 4: Print the updated grouped indices
for group_name, indices in grouped_indices.items():
    print(f"{group_name}: {indices}")









################################################################################
###         Performing the ADF test and grouping the indexes by type         ###
################################################################################

import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Step 1: Load the compact dataset with a DateTime index
df_compact = pd.read_csv("compact_dataset.csv", index_col=0, parse_dates=True)

# Step 2: Initialize dictionaries to store stationary and non-stationary series
stationary_indices = []
stationary_labels = []
non_stationary_indices = []
non_stationary_labels = []

# Step 3: Perform the Augmented Dickey-Fuller test on each column
for idx, column in enumerate(df_compact.columns):
    # Perform ADF test on each column (dropna to handle missing values)
    result = adfuller(df_compact[column].dropna())  # dropna to handle missing values

    # Get the p-value from the result
    p_value = result[1]

    # Classify as stationary or non-stationary based on p-value
    if p_value < 0.05:
        stationary_indices.append(idx)
        stationary_labels.append(column)
    else:
        non_stationary_indices.append(idx)
        non_stationary_labels.append(column)

# Step 4: Return the stationary and non-stationary series by index and label
print("Stationary series:")
for idx, label in zip(stationary_indices, stationary_labels):
    print(f"Index: {idx}, Label: {label}")

print("\nNon-stationary series:")
for idx, label in zip(non_stationary_indices, non_stationary_labels):
    print(f"Index: {idx}, Label: {label}")

### After running this method, we confirm that all the series are indeed stationary

#### Possible logic for handling non-stationary series (not necessary for the "compact" dataset) ####



# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the compact dataset from CSV, ensuring the time index is recognized
df_compact = pd.read_csv("compact_dataset.csv", index_col=0, parse_dates=True)

# Step 2: Standardize the numeric data (ignoring the index and any non-numeric columns)
# Select only the numeric columns (ignore the index and non-numeric columns if any)
numeric_columns = df_compact.select_dtypes(include=['float64', 'int64']).columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Step 3: Standardize the numeric columns while preserving the time index
df_standardized = df_compact.copy()  # Copy the original dataframe to preserve it
df_standardized[numeric_columns] = scaler.fit_transform(df_compact[numeric_columns])

# Step 4: Optionally, save the standardized data to a new CSV file, preserving the time index
df_standardized.to_csv("standardized_compact_dataset.csv")

# Step 5: Print out the first few rows to check the results, including the time index
print(df_standardized.head())


# #### Soybilgen and Yazgan, pp. 391-392

# #### Kalman update diag method (is required by the Kalman filter diag  method)

# In[ ]:


# run_cell = False  # Change to True to run this block

import os
import numpy as np
from numpy.linalg import pinv, det

def kalman_update_diag(A, C, Q, R, y, x, V, initial):
    """
    Perform a one-step Kalman filter update.

    Parameters:
    A, C, Q, R : np.array
        System and observation matrices and covariances.
    y : np.array
        Observation vector for current time step.
    x : np.array
        Prior state mean estimate.
    V : np.array
        Prior covariance estimate.
    initial : bool
        Whether this is the initial step.

    Returns:
    dict
        xnew : np.array
            Updated state estimate.
        Vnew : np.array
            Updated covariance estimate.
        VVnew : np.array
            Cross-covariance estimate.
        loglik : float
            Log-likelihood of the current observation.
    """
    ss = A.shape[0]  # State size

    # Prediction step
    if initial:
        xpred = x
        Vpred = V
    else:
        xpred = A @ x
        Vpred = A @ V @ A.T + Q

    # Innovation
    e = y - C @ xpred
    S = C @ Vpred @ C.T + R
    Sinv = np.linalg.inv(S)

    # Log-likelihood calculation
    detS = det(S)
    loglik = -0.5 * (np.log(detS) + e.T @ Sinv @ e + len(e) * np.log(2 * np.pi))

    # Kalman gain
    K = Vpred @ C.T @ Sinv

    # State and covariance update
    xnew = xpred + K @ e
    Vnew = (np.eye(ss) - K @ C) @ Vpred

    # Cross-covariance
    VVnew = (np.eye(ss) - K @ C) @ A @ V

    return {"xnew": xnew, "Vnew": Vnew, "VVnew": VVnew, "loglik": loglik.item()
    }




# #### Kalman filter diag method (requires Kalman update diag method; is required by Kalman smoother diag method)

# In[ ]:


# run_cell = False  # Change to True to run this block

import os

import numpy as np


def kalman_filter_diag(y, A, C, Q, R, init_x, init_V, model):
    """
    Kalman filter implementation.

    Parameters:
    y : np.array
        Observations matrix of shape (N, T).
    A : np.array
        State transition matrices of shape (ss, ss, T).
    C : np.array
        Observation matrices of shape (N, ss, T).
    Q : np.array
        Process noise covariance matrices of shape (ss, ss, T).
    R : np.array
        Measurement noise covariance matrices of shape (N, N, T).
    init_x : np.array
        Initial state estimate of shape (ss,).
    init_V : np.array
        Initial covariance estimate of shape (ss, ss).
    model : list or range
        Sequence of model indices for each time step.

    Returns:
    dict
        x : np.array
            Filtered state estimates of shape (ss, T).
        V : np.array
            Filtered state covariances of shape (ss, ss, T).
        VV : np.array
            Cross-covariances of shape (ss, ss, T).
    """
    os = y.shape[0]  # Number of observations
    T = y.shape[1]  # Number of time steps
    ss = A.shape[0]  # State space size

    x = np.zeros((ss, T))
    V = np.zeros((ss, ss, T))
    VV = np.zeros((ss, ss, T))
    loglik = 0

    for t in range(T):
        m = model[t]
        if t == 0:
            prevx = init_x
            prevV = init_V
            initial = True
        else:
            prevx = x[:, t-1].reshape(-1, 1)
            prevV = V[:, :, t-1]
            initial = False

        result_kud = kalman_update_diag(
            A[:, :, m-1], C[:, :, m-1], Q[:, :, m-1], R[:, :, m-1],
            y[:, t].reshape(-1, 1), prevx, prevV, initial
        )

        x[:, t] = result_kud["xnew"].flatten()
        V[:, :, t] = result_kud["Vnew"]
        VV[:, :, t] = result_kud["VVnew"]
        loglik += result_kud["loglik"]

    return {"x": x, "V": V, "VV": VV}



# #### Smooth update method (is required by Kalman smoother diag)

# In[ ]:


# run_cell = False  # Change to True to run this block

import os

# Check if the file already exists

import numpy as np
from numpy.linalg import pinv

def smooth_update(xsmooth_future, Vsmooth_future, xfilt, Vfilt, Vfilt_future, VVfilt_future, A, Q):
    """
    Perform one step of the backwards RTS smoothing equations.

    Parameters:
    xsmooth_future : np.array
        E[X_t+1|T], Smoothed state estimate at t+1.
    Vsmooth_future : np.array
        Cov[X_t+1|T], Smoothed covariance at t+1.
    xfilt : np.array
        E[X_t|t], Filtered state estimate at time t.
    Vfilt : np.array
        Cov[X_t|t], Filtered covariance at time t.
    Vfilt_future : np.array
        Cov[X_t+1|t+1], Filtered covariance at time t+1.
    VVfilt_future : np.array
        Cov[X_t+1, X_t|t+1], Cross-covariance at time t+1.
    A : np.array
        State transition matrix at time t+1.
    Q : np.array
        Process noise covariance at time t+1.

    Returns:
    dict
        xsmooth : np.array
            E[X_t|T], Smoothed state estimate at time t.
        Vsmooth : np.array
            Cov[X_t|T], Smoothed covariance at time t.
    """
    # Prediction step
    xpred = A @ xfilt
    Vpred = A @ Vfilt @ A.T + Q

    # Smoother gain matrix
    J = Vfilt @ A.T @ pinv(Vpred)

    # Smoothed estimates
    xsmooth = xfilt + J @ (xsmooth_future - xpred)
    Vsmooth = Vfilt + J @ (Vsmooth_future - Vpred) @ J.T

    return {"xsmooth": xsmooth, "Vsmooth": Vsmooth}





# #### Kalman smoother diag (requires Kalman filter diag and Smoother update; is required by Factor extraction)

# In[ ]:


import os

import numpy as np
# from smooth_update import smooth_update
# from kalman_filter_diag import kalman_filter_diag


def kalman_smoother_diag(y, A, C, Q, R, init_x, init_V, model):
    """
    Custom implementation of the Kalman smoother for a time-varying state-space model.

    Parameters:
    y : np.array
        Observations matrix of shape (N, T) where N is the number of observed variables and T is the number of time steps.
    A : np.array
        State transition matrices of shape (ss, ss, T).
    C : np.array
        Observation matrices of shape (N, ss, T).
    Q : np.array
        Process noise covariance matrices of shape (ss, ss, T).
    R : np.array
        Measurement noise covariance matrices of shape (N, N, T).
    init_x : np.array
        Initial state estimate of shape (ss,).
    init_V : np.array
        Initial covariance estimate of shape (ss, ss).
    model : list or range
        Sequence of model time steps, typically range(1, T+1).

    Returns:
    dict
        Contains xsmooth (smoothed state estimates) and Vsmooth (smoothed covariance estimates).
    """
    T = y.shape[1]
    ss = A.shape[0]

    # Initialize smoothed state and covariance arrays
    xsmooth = np.zeros((ss, T))
    Vsmooth = np.zeros((ss, ss, T))

    # Forward pass: Run the Kalman filter
    kfd_result = kalman_filter_diag(y, A, C, Q, R, init_x, init_V, model)
    xfilt = kfd_result["x"]
    Vfilt = kfd_result["V"]
    VVfilt = kfd_result["VV"]

    # Backward pass: Run the RTS smoother
    xsmooth[:, T-1] = xfilt[:, T-1]
    Vsmooth[:, :, T-1] = Vfilt[:, :, T-1]

    for t in range(T-2, -1, -1):
        m = model[t+1]
        result_s_update = smooth_update(
            xsmooth[:, t+1].reshape(-1, 1),
            Vsmooth[:, :, t+1],
            xfilt[:, t].reshape(-1, 1),
            Vfilt[:, :, t],
            Vfilt[:, :, t+1],
            VVfilt[:, :, t+1],
            A[:, :, m-1],
            Q[:, :, m-1]
        )

        xsmooth[:, t] = result_s_update["xsmooth"].flatten()
        Vsmooth[:, :, t] = result_s_update["Vsmooth"]

    return {"xsmooth": xsmooth, "Vsmooth": Vsmooth}



# #### RicSW (is required by Factor extraction)

# In[ ]:


### Testing code

import numpy as np
from scipy.linalg import eig, pinv, block_diag
from numpy.linalg import inv

x = standardized_df.values  # Assuming standardized_df is a DataFrame
T, N = x.shape

# Output the values of T, N, and the first rows of x
print("T (Number of rows):", T)
print("N (Number of columns):", N)
print("First 5 rows and 5 columns of x:\n", x[:5, :5])

r = 2
q = 2
p = 1
nlag = p - 1

A_temp = np.zeros((r, r * p)) # Creates a temporary matrix filled with zeros of dimension r x r*p
I = np.eye(r * p) # Creates an indentiy matrix of size r*p x r*p

print(type(A_temp))
print("Size (number of elements):", A_temp.size)
print("Shape (dimensions):", A_temp.shape)
print("Number of dimensions:", A_temp.ndim)
print(A_temp[:2])

print(type(I))
LL = I.shape[0]
if p != 1:
    A = np.vstack((A_temp.T, I[:LL-r, :]))  # Equivalent to rbind(A_temp, I[1:(LL-r), ])
else:
    A = np.vstack((A_temp.T, np.empty((0, r * p))))  # Equivalent to rbind(t(A_temp), I[0, ]) # Modified due to different indexing between R and Python


print("Size (number of elements):", A.size)
print("Shape (dimensions):", A.shape)
print("Number of dimensions:", A.ndim)
print(A[:3]) # 2 x 2 mtrix of zeros

Q = np.zeros((r * p, r * p)) # 2 x 2 identiy matrix
Q[:r, :r] = np.eye(r)
print("Size (number of elements):", Q.size)
print("Shape (dimensions):", Q.shape)
print("Number of dimensions:", Q.ndim)
print(Q[:3])

cov_x = np.cov(x, rowvar=False)
print("Size (number of elements):", cov_x.size)
print("Shape (dimensions):", cov_x.shape)
print("Number of dimensions:", cov_x.ndim)
print(cov_x[:3])
eigvals, eigvecs = eig(cov_x)
# Print eigenvalues
print("Eigenvalues:")
print(eigvals)
print("Size (number of elements):", eigvals.size)

# Print eigenvectors
print("\nEigenvectors:")
print(eigvecs)
print("Size (number of elements):", eigvecs.shape)


idx = eigvals.argsort()[::-1]
print(idx)

eigvals, eigvecs = eigvals[idx][:r], eigvecs[:, idx][:, :r]
print(eigvals.shape)
print(eigvecs.shape)

F = x @ eigvecs
print(x.size)
print(x.shape)
print(F.shape)
print(F) # 379 x 2 matrix

R = np.diag(np.diag(np.cov(x - F @ eigvecs.T, rowvar=False)))
print(R)
print(R.shape) # 118 x 118 matrix


# VAR model estimation
Z = F[:-1, :]
z = F[1:, :]
print(Z.shape)
print(Z)
print(z)
print(z.shape)


A_temp = inv(Z.T @ Z) @ Z.T @ z
A[:r, :r * p] = A_temp.T
print(A_temp.shape)

e = z - Z @ A_temp  # VAR residuals
print(e.shape)
print(e)
H = np.cov(e, rowvar=False)
print(H.shape)

Q[:r, :r] = H
print(Q.shape)
print(Q)

#### Execute until here ####
print(A.shape)

initx = F[0, :]
kron_A = np.kron(A, A)
print(kron_A.shape)
print(kron_A)

matrix_1 = np.eye(r * p**2)
print(matrix_1.shape)
print(matrix_1)

Q_flatten = Q.flatten(order='F').reshape(-1, 1)
print(Q_flatten.shape)

diag_matrix = np.eye(kron_A.shape[0])
matrix_2 = pinv(diag_matrix - kron_A)
print(matrix_2.shape)

initV = pinv(diag_matrix - kron_A) @ Q.flatten(order='F').reshape(-1, 1)
print(initV.shape)
initV = initV.reshape((r * p, r * p), order='F')
print(initV)
C = np.hstack((eigvecs, np.zeros((N, r * nlag))))
print(C)

print(C.shape)


# In[ ]:


import numpy as np

from scipy.linalg import eig, pinv, block_diag
from numpy.linalg import inv

def ricSW(standardized_df, q, r, p):
    """
    Computes parameters for a factor model using standardized data.

    Parameters:
    standardized_df : np.array
        Standardized and balanced panel data of size
    q : int
        Rank for reduced Q covariance matrix (if applicable).
    r : int
        Number of factors.
    p : int
        Lag order for VAR.

    Returns:
    dict
        A dictionary containing factor model parameters.
    """
    x = standardized_df.values  # Assuming standardized_df is a DataFrame
    T, N = x.shape # T:number of rows N: number of columns
    nlag = p - 1  # Order of lags in the VAR model for the factors. Typically zero if p=1 (number of additional lags beyond t-1)

    # Companion form initialization
    # State-space representation: In a dynamic factor model, factors evolve according to a VAR process.
    # This requires representing the state transitions and noise in a specific structured form: xt​=Axt−1​+εt​, εt​∼N(0,Q)
    A_temp = np.zeros((r, r * p)) # Initializes A, the state transition matrix, to store the VAR coefficients for the first r variables (factors) across p lags.
    # Initially, this is all zeros. Dimensions r=number of factors; r*p= total number of variables in the companion form, accounting for p lags.
    I = np.eye(r * p) # Creates an identity matrix of size r*p x r*p
    # Logic for the creation of the companion matrix A, which represents how the factors evolve over time
    if p != 1:
        A = np.vstack((A_temp, I[:-r, :])) # Matrix obtained by stacking the matrix of the VAR coefficient and the identity matrix that shifts lagged factors (slice of the identity matrix). Size r*(p−1) x r*p
    else:
        A = np.vstack((A_temp.T, np.zeros((1, r * p))))

    Q = np.zeros((r * p, r * p)) # Initializes a matrix Q of zeros
    Q[:r, :r] = np.eye(r) # Fills the top left of the matrix Q with an identity matrix

    # Compute eigenvalues and eigenvectors of the covariance matrix
    cov_x = np.cov(x, rowvar=False) # Computing the covariance of the data in x. With Rowvar = False, we are calculating the covariance between columns (features)
    eigvals, eigvecs = eig(cov_x) # Computing the eigenvalues and eigenvectors of the covariance matrix
    idx = eigvals.argsort()[::-1]  # Sort eigenvalues in descending order
    eigvals, eigvecs = eigvals[idx][:r], eigvecs[:, idx][:, :r] # Selecting the top r eigenvalues and their corresponding eigenvectors

    # Principal component estimates
    F = x @ eigvecs # Transforms the original data x into a new space defined by the eigenvectors.
    # Essentially, it projects the data onto the new axes (principal components) defined by eigvecs. The matrix F now represents the data in the reduced principal component space.
    R = np.diag(np.diag(np.cov(x - F @ eigvecs.T, rowvar=False)))
    # 1) eigvecs.T is the transpose of the matrix of eigenvectors. The matrix multiplication F @ eigvecs.T reconstructs the approximation of the original data points from their projections (F),
    # by multiplying the transformed data (F) by the eigenvectors. F @ eigvecs.T brings the data back to the original feature space (though the approximation may have some loss due to dimensionality reduction).
    # 2) x - F @ eigvecs.T computes the difference between the original data x and the approximation of the data obtained by projecting it back to the original feature space.
    # This difference represents the reconstruction error or the residuals between the original data and the approximated data.
    # 3) np.cov() computes the covariance matrix of the reconstruction error
    # 4) np.diag(np.diag(...)) takes the covariance matrix of the residuals and extracts the diagonal elements
    # In brief, R tells us how much variance remains in the data for each feature after projecting it onto the principal component space defined by eigvecs.
    # It captures the variance that was not explained by the selected principal components.

    # VAR model estimation
    if p == 1:
    # For VAR(1), we just need one lag: F_{t-1}
        Z = F[:-1, :]         # Lagged values: F_{t-1}. Z contains the lagged values of the state vector F (the variables we're modeling).
        z = F[1:, :]          # Current values: F_t
    else:
    # For VAR(p), we need p lags: F_{t-1}, F_{t-2}, ..., F_{t-p}
        Z = np.hstack([F[p - kk - 1:-(kk + 1), :] for kk in range(p)])  # Stack lags F_{t-1}, F_{t-2}, ..., F_{t-p}
        z = F[p:, :]          # Current values: F_t


    # 1) For p=1 (VAR(1)):
    # We only need the first lag, Ft−1​, which is the immediate previous value of F.
    # The design matrix Z is simply F[:−1,:], which takes all rows except the last (i.e., the lagged values).
    # The response matrix Ft is F[1:,:], which contains the current values of F starting from the second row.

    # 2) For p>1 (VAR(p)):
    # We need to include multiple lags: Ft−1,Ft−2,…,Ft−p​.
    # The design matrix Z is built by horizontally stacking the lagged values of F from Ft−1 to Ft−p​ using np.hstack.
    # The response matrix Ft​ is built by taking all rows starting from index p (i.e., excluding the first p rows).
    # This is the case when we have a higher-order VAR model, where each observation depends on several past observations.
    # This is the standard case for a VAR(1) model where each observation depends only on the previous observation.

    # 3) Result: Z is a matrix of shape (n_samples - nlag, p * n_features) that combines the p previous lags (time steps) of data from F.
    # This matrix Z forms the "design matrix" used in VAR estimation.

    A_temp = inv(Z.T @ Z) @ Z.T @ z
    # performs the OLS estimation for the VAR model. It estimates the coefficients (weights) of the model that relate the lagged values (from the design matrix Z) to the current values (in z)
    A[:r, :r * p] = A_temp.T
    # This places the transposed estimated coefficients A_temp.T into the matrix A.
    # The result is that the coefficient matrix A is populated with the estimated values for the VAR model from A_temp.

    # Compute Q
    e = z - Z @ A_temp  # VAR residuals
    # Residuals represent the differences between the observed values (z) and the predicted values (Z @ A_temp).
    # These residuals capture the error in the VAR model's predictions.
    # The residuals, e, represent the part of the observed data that cannot be explained by the lagged values of the series.
    H = np.cov(e, rowvar=False) # Covariance matrix of the residuals e.
    # This matrix, H, gives an estimate of how the errors in the VAR model are related to each other.
    # Specifically, it tells you whether the residuals (or errors) across different variables are correlated with each other.

    if r == q:
        Q[:r, :r] = H # H is the covariance matrix of residuals, and the top-left r x r block of Q is now populated with the values from H. The rest of Q remains zero.
        # r=q, there is no need to perform eigenvalue decomposition to extract specific components because we are considering the entire variance structure of H.
        # The matrix Q is directly updated with H, as H already fully represents the covariance structure.
    else:
        eigvals_H, eigvecs_H = eig(H)
        idx = eigvals_H.argsort()[::-1][:q]  # Sorts the eigenvalues in descending order ([::-1]) and selects the top q eigenvalues.
        eigvals_H, eigvecs_H = eigvals_H[idx], eigvecs_H[:, idx] # Updates the eigenvalues and eigenvectors arrays to keep only the top q eigenvalues and their corresponding eigenvectors.
        Q[:r, :r] = eigvecs_H @ np.diag(eigvals_H) @ eigvecs_H.T # Updates the top-left r x r block of Q by replacing it with a matrix computed from the eigenvectors and eigenvalues.

    # 1) np.diag(eigvals_H) : Creates a diagonal matrix from the selected eigenvalues.
    # 2) eigvecs_H @ np.diag(eigvals_H) @ eigvecs_H.T : Forms the diagonalized covariance matrix using the top q eigenvalues and corresponding eigenvectors.
    # If q < r, we only want to retain a reduced-rank approximation of H that captures the most important variance information (top q eigenvalues and their corresponding eigenvectors).
    # The covariance matrix HH may contain noise or redundant information in its lower-ranked eigenvalues and eigenvectors.
    # By retaining only the top q components, we simplify H while preserving the most significant variance contributions.

    # Initialize Kalman filter parameters
    initx = F[0, :]
    # Selects the first row of F, corresponding to the latent representation of the first time step in the reduced space.
    # initx serves as the initial state for the system in the latent r-dimensional space.

    kron_A = np.kron(A, A)
    # The Kronecker product models pairwise interactions between the coefficients of A.
    # Each element of A is expanded into a block of size r×(r⋅p), which reflects all possible combinations of coefficients in A.
    initV = pinv(np.eye(r * p**2) - kron_A) @ Q.flatten(order='F')
    # 1) Subtracting kron_A from the identity matrix representing a transformation or propagation operator in the context of covariance dynamics
    # 2) The Moore-Penrose pseudoinverse is computed for the matrix
    # 3) Reshaping Q into a 1D array by stacking its columns sequentially (column-major order, Fortran-style). Why?
    # The vectorized form of Q is required because the system equation, after vectorization, operates on 1D arrays rather than 2D matrices.
    # Specifically, the vectorized form of the Lyapunov equation for the steady-state covariance matrix Σ (initV),
    # which encodes how noise from Q propagates through the dynamics of the system, represented by A, to create long-run variances and covariances of the system's state variables.
    initV = initV.reshape((r * p, r * p), order='F')
    #  steady-state covariance matrix of the VAR system's state vector, capturing variances and covariances of all variables and their lags in the long run,
    # reshaped from vectorized to matrix form. Matrix of dimensions r*p x r*p

    C = np.hstack((eigvecs, np.zeros((N, r * nlag))))
    # This code constructs a matrix C by horizontally stacking two components:
    # 1) eigvecs: a matrix of size N×rN×r, containing eigenvectors derived earlier (likely capturing the system's dominant modes or components)
    # 2) np.zeros((N, r * nlag)): a matrix of zeros of size N×(r⋅nlag), added as padding.
    # C therefore combines the contributions from the current state and the lagged terms, with the lagged terms initially set to zero.

    return {
        "A": A, "C": C, "Q": Q, "R": R, "initx": initx, "initV": initV
    }





# #### Factor extraction (main); requires RicSW and Kalman smoother diag

# In[ ]:


# %run kalman_update_diag.py
# %run kalman_filter_diag.py
# %run smooth_update.py
# %run kalman_smoother_diag.py
# %run ricSW.py

# import os
# print(os.getcwd())


# Integrate kalman_smoother_diag into the factor_extraction function
# from ricSW import ricSW
# from kalman_smoother_diag import kalman_smoother_diag
def factor_extraction(standardized_df, q, r, p):
    """
    Extracts common factors from a balanced, standardized dataset.
    """
    if r < q:
        raise ValueError("q must be less than or equal to r.")
    if p < 1:
        raise ValueError("p must be greater than or equal to 1.")

    # Step 1: Use ricSW to estimate parameters
    result_ricsw = ricSW(standardized_df, q, r, p)
    A, C, Q, R = result_ricsw["A"], result_ricsw["C"], result_ricsw["Q"], result_ricsw["R"]
    initx, initV = result_ricsw["initx"], result_ricsw["initV"]

    # Step 2: Kalman filter initializations
    T = standardized_df.shape[0]
    AA = np.repeat(A[:, :, np.newaxis], T, axis=2)
    QQ = np.repeat(Q[:, :, np.newaxis], T, axis=2)
    CC = np.repeat(C[:, :, np.newaxis], T, axis=2)
    RR = np.repeat(R[:, :, np.newaxis], T, axis=2)

    # Handle missing data by assigning large noise variance to missing values
    for jt in range(T):
        miss = np.isnan(standardized_df.iloc[jt, :].values)
        Rtemp = np.diag(R)
        Rtemp[miss] = 1e+32
        RR[:, :, jt] = np.diag(Rtemp)

    xx = standardized_df.values
    xx[np.isnan(xx)] = 0

    # Step 3: Run the Kalman smoother
    model = range(1, T+1)
    smoother_result = kalman_smoother_diag(xx.T, AA, CC, QQ, RR, initx, initV, model)

    F = smoother_result["xsmooth"].T
    VF = smoother_result["Vsmooth"]

    return {"F": F, "VF": VF, "A": A, "C": C, "Q": Q, "R": R, "initx": initx, "initV": initV}


standardized_df.info()



# ##### End session and restart kernel

# In[ ]:


# run_cell = False  # Change to True to run this block

# import os
# import shutil

# cwd = os.getcwd()
# for filename in os.listdir(cwd):
#     file_path = os.path.join(cwd, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)  # Remove file or link
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)  # Remove directory
#     except Exception as e:
#         print(f'Failed to delete {file_path}. Reason: {e}')


# !pwd


