'''
Module: test_files_n_folders.py
Description:
    the module checks the existance of the home folder structure as per named constants used in project
'''

import os

from src.files_n_folders import *

def test_files_n_folders_f():
    assert type(files_n_folders_f()) is dict

    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       DATA_FOLDER
                        ))
    
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       MODELS_FOLDER
                        ))
    
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       MODELS_FOLDER,
                                       PREPROC_FOLDER
                        ))
    
    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       CONFIG_FOLDER
                        ))

    assert os.path.exists(os.path.join(
                                       PROJECT_FOLDER, 
                                       DATA_FOLDER, 
                                       DATASET_NAME
                        ))