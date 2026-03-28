import os
import random

import pandas as pd
import requests

from src.files_n_folders import PROJECT_FOLDER, DATA_FOLDER, DATASET_NAME

# Load the heart.csv data as a source for test-cases for the REST API
data_file_full_name = os.path.join(
        PROJECT_FOLDER, 
        DATA_FOLDER, 
        DATASET_NAME
    )
ddf = pd.read_csv(data_file_full_name)

# Select random row where target == 1
idx_1 = random.choice(ddf[ddf['target'] == 1].index)  # Randomly select an index where target == 1
row_target_1 = ddf.iloc[idx_1, :-1].to_dict()  # Convert to dictionary for JSON serialization
for k in row_target_1.keys():
    if k != 'oldpeak':
        row_target_1[k] = int(row_target_1[k])  # Convert to int for JSON serialization 

# Select random row where target == 0
idx_0 = random.choice(ddf[ddf['target'] == 0].index)  # Randomly select an index where target == 0
row_target_0 = ddf.iloc[idx_0, :-1].to_dict()  # Convert to dictionary for JSON serialization
for k in row_target_0.keys():
    if k != 'oldpeak':
        row_target_0[k] = int(row_target_0[k])  # Convert to int for JSON serialization

# API endpoint
url = 'http://localhost:5000/submit_analysis'

# Test 0: Check if the API is reachable first
try:
    response = requests.get(url)
    print("REST API server is reachable. Status code:", response.status_code)
    print('Proceeding with ML-model testing...')
    print()
except:
    print("Error occurred while checking API reachability:")
    print("Please check if the REST API server is running at the specified URL:", url)
    os._exit(1)

# Test 1: Send data for target == 1
response_1 = requests.post(url, json=row_target_1)
print("Test for patient marked as having cancer:")
print("Request data simulated from the randomly selected row no.", idx_1, ":", row_target_1)
print("Response:", response_1.json())
print()

# Test 2: Send data for target == 0
response_0 = requests.post(url, json=row_target_0)
print("Test for patient marked as not having cancer:")
print("Request data simulated from the randomly selected row no.", idx_0, ":", row_target_0)
print("Response:", response_0.json())
