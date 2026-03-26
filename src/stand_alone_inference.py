'''
Module: stand_alone_inference.py
Description:
    the module defines stand alone prediction functions for offline inference usage
    e.g.: used by REST API server publishing the trained model as a service
'''
from json import encoder
import os
import numpy as np
import pandas as pd

from joblib import load as load_fld_handler

from src.files_n_folders import PROJECT_FOLDER, MODELS_FOLDER, PREPROC_FOLDER
from src.ml_model_train_n_save import MODEL_SHORT_NAME

def load_model_n_predict(new_data: pd.DataFrame) -> np.array:
    '''
    loads the offline saved model (pre-passing training pipeline is supposed)
    and returns model predictions on the new_data dataset passed as parameter
    '''

    model_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER, 
        'model_save_' + MODEL_SHORT_NAME + '.bin'
        )

    model = load_fld_handler(model_save_file_name)

    scaler_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'num_flds_scaler_save.bin'
   )

    scaler = load_fld_handler(scaler_save_file_name)

    encoder_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'cat_flds_encoder_save.bin'
   )

    encoder = load_fld_handler(encoder_save_file_name)

    fields_type_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER,
        PREPROC_FOLDER, 
        'flds_types_save.bin'
    )

    flds_types_dict = load_fld_handler(fields_type_save_file_name)

    num_flds = flds_types_dict['num_flds']
    cat_flds = flds_types_dict['cat_flds']  
    bin_flds = flds_types_dict['bin_flds'][:-1]
    X_num = scaler.transform(new_data[num_flds])
    X_cat = encoder.transform(new_data[cat_flds])
    X_bin = new_data[bin_flds].values
    X_preproc = np.concatenate([X_num, X_cat, X_bin], axis=1)

    return model.predict(X_preproc)