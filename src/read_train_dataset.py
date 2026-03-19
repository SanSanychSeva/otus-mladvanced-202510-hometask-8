'''
Module: read_train_dataset.py
Description:
    the module contains the function to read the training dataset in the project folders structure
'''
import os

import pandas as pd

from src.files_n_folders import DATA_FOLDER, DATASET_NAME, PROJECT_FOLDER

def read_train_dataset_f() -> pd.DataFrame:
    '''
    Reads and returns the dataset from the path, 
    whose components are specified in the files_n_folders module
    '''

    data_file_full_name = os.path.join(
        PROJECT_FOLDER, 
        DATA_FOLDER, 
        DATASET_NAME
    )

    ddf = pd.read_csv(data_file_full_name)
    return ddf