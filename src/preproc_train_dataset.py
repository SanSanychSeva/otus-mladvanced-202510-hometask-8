'''
Module: preproc_train_dataset.py
Description:
    module for pre-processing the training dataset, to be used after the exploratory data analysis
'''

import os
from joblib import dump as save_fld_handler

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from src.files_n_folders import PROJECT_FOLDER, MODELS_FOLDER, PREPROC_FOLDER

def preproc_num_fields_f(df:'pd.DataFrame', num_flds:list) -> 'np.ndarray':
    '''
    transforms the numerical fields by standardizing them and saves the scaler for later use on the test dataset
    '''
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_flds])

    scaler_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'num_flds_scaler_save.bin'
   )

    save_fld_handler(scaler, scaler_save_file_name, compress=True)
    return X_num

def preproc_cat_fields_f(df:'pd.DataFrame', cat_flds:list) -> 'np.ndarray':
    '''
    transforms the categorical fields by one-hot encoding them and saves the encoder for later use on the test dataset
    '''
    encoder = OneHotEncoder(sparse_output=False)
    X_cat = encoder.fit_transform(df[cat_flds])

    encoder_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'cat_flds_encoder_save.bin'
   )

    save_fld_handler(encoder, encoder_save_file_name, compress=True)
    return X_cat

def preproc_dataset_f(df:'pd.DataFrame',
                        num_flds:list,
                        cat_flds:list,
                        bin_flds:list) -> 'np.ndarray':
    '''
    pre-processes the training dataset by transforming the numerical fields and the categorical fields 
    and returns the transformed reassambled dataset as a numpy array (last column being the target field)
    '''
    X_num = preproc_num_fields_f(df, num_flds)
    X_cat = preproc_cat_fields_f(df, cat_flds)
    X_bin = df[bin_flds].values

    fields_type_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'flds_types_save.bin'
   )

    flds_types_dict = {
        'num_flds': num_flds,
        'cat_flds': cat_flds,
        'bin_flds': bin_flds
    }

    save_fld_handler(flds_types_dict, fields_type_save_file_name, compress=True)

    return np.concatenate([X_num, X_cat, X_bin], axis=1)