'''
Module: test_files_n_folders.py
Description:
    the module checks the existance of the home folder structure as per named constants used in project
'''

import os

from src.files_n_folders import MODELS_FOLDER, PREPROC_FOLDER, PROJECT_FOLDER
from src.files_n_folders import CONFIG_FOLDER, DATASET_NAME, DATA_FOLDER

def check_data_folder():
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       DATA_FOLDER
                        ))
    
def check_model_folder():
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       MODELS_FOLDER
                        ))
    
def check_preproc_folder():
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       PREPROC_FOLDER
                        ))
    
def check_config_folder():
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       CONFIG_FOLDER
                        ))

def check_dataset():
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       DATA_FOLDER, 
                                       DATASET_NAME
                        ))