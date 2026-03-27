import os

import pandas as pd
import requests
from flask import jsonify

from src.files_n_folders import PROJECT_FOLDER, DATA_FOLDER, DATASET_NAME

# Load the heart.csv data
data_file_full_name = os.path.join(
        PROJECT_FOLDER, 
        DATA_FOLDER, 
        DATASET_NAME
    )
ddf = pd.read_csv(data_file_full_name)
print(ddf.info())

# Select one row where target == 1
row_target_1 = ddf[ddf['target'] == 1].iloc[0, :-1].to_dict()  # Convert to dictionary for JSON serialization
for k in row_target_1.keys():
    if k != 'oldpeak':
        row_target_1[k] = int(row_target_1[k])  # Convert to int for JSON serialization 

# Select one row where target == 0
row_target_0 = ddf[ddf['target'] == 0].iloc[0, :-1].to_dict()  # Convert to dictionary for JSON serialization
for k in row_target_0.keys():
    if k != 'oldpeak':
        row_target_0[k] = int(row_target_0[k])  # Convert to int for JSON serialization

# API endpoint
url = 'http://localhost:5000/submit_analysis'

# Test 1: Send data for target == 1
response_1 = requests.post(url, json=row_target_1)
print("Test for target == 1:")
print("Request data:", row_target_1)
print("Response:", response_1.json())
print()

# Test 2: Send data for target == 0
response_0 = requests.post(url, json=row_target_0)
print("Test for target == 0:")
print("Request data:", row_target_0)
print("Response:", response_0.json())
