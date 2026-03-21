'''
This module contains the tests for the pre-processing of the training dataset
'''

import os
from joblib import load as load_fld_handler

import numpy as np
import pandas as pd

from src.files_n_folders import PROJECT_FOLDER, MODELS_FOLDER, PREPROC_FOLDER
from src.read_train_dataset import read_train_dataset_f
from src.preproc_train_dataset import preproc_num_fields_f, preproc_cat_fields_f
from src.eda_train_dataset import run_eda_train_dataset_f
from src.preproc_train_dataset import preproc_dataset_f

def test_all_preproc_f():
    '''
    tests the pre-processing of the training dataset by loading the saved scaler and encoder 
    and inversely applying them to compare the results with the initial dataset parts
    '''

    ddf = read_train_dataset_f()
    num_flds, cat_flds, bin_flds = run_eda_train_dataset_f(ddf)
    scaler_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'num_flds_scaler_save.bin'
   )
    encoder_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'cat_flds_encoder_save.bin'
   )

    scaler_saved = load_fld_handler(scaler_save_file_name)
    encoder_saved = load_fld_handler(encoder_save_file_name)

    ddf_num_expected = scaler_saved.inverse_transform(preproc_num_fields_f(ddf, num_flds))
    ddf_cat_expected = encoder_saved.inverse_transform(preproc_cat_fields_f(ddf, cat_flds))

    assert np.allclose(ddf_num_expected, ddf[num_flds].values)
    assert (ddf_cat_expected != ddf[cat_flds].values).sum() == 0
    assert (preproc_dataset_f(ddf, num_flds, cat_flds, bin_flds)[:,-len(bin_flds):] != 
            ddf[bin_flds].values).sum() == 0