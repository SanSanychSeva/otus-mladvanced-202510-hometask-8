'''
Module: test_ml_model_train_n_save.py
Description:
    This module contains a test function for the ml_model_train_n_save_f function.
'''

import os
from joblib import load as load_fld_handler

import numpy as np

from src.files_n_folders import PROJECT_FOLDER, MODELS_FOLDER, DATA_FOLDER
from src.preproc_train_dataset import preproc_dataset_f
from src.save_validation_data import save_validation_data_f
from src.read_train_dataset import read_train_dataset_f
from src.eda_train_dataset import run_eda_train_dataset_f
from src.ml_model_train_n_save import ml_model_train_n_save_f, MODEL_SHORT_NAME

def test_ml_model_train_n_save_f():
    ddf = read_train_dataset_f()
    num_flds, cat_flds, num_cat_flds = run_eda_train_dataset_f(ddf)
    Xy = preproc_dataset_f(ddf, num_flds, cat_flds, num_cat_flds)
    X_train, y_train = save_validation_data_f(Xy)
    model_new = ml_model_train_n_save_f(X_train, y_train)

    model_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER, 
        'model_save_' + MODEL_SHORT_NAME + '.bin'
    )
    model_saved = load_fld_handler(model_save_file_name)
    
    Xy_val = np.load(os.path.join(PROJECT_FOLDER, DATA_FOLDER, 'val_Xy_save.npy'))
    X_val = Xy_val[:,:-1]
    y_val = Xy_val[:,-1]

    assert np.isclose(model_new.score(X_val, y_val), model_saved.score(X_val, y_val))