# %%
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