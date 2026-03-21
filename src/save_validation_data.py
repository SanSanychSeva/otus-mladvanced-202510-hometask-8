'''
Module: save_validation_data.py
Description:
    the module contains splits dataset into train and val sets, 
    saving the validation data and returning the training dataset
'''
import os
import numpy as np

from sklearn.model_selection import train_test_split

from src.files_n_folders import DATA_FOLDER, PROJECT_FOLDER

def keep_validation_data_f(Xy:'np.ndarray', val_size:float=0.2) -> tuple:
    """   
    splits the Xy-dataset into X and y train and val sets, 
    saving the validation data to data folder and 
    """
    # Calculate the number of samples to keep as validation data
    X_train, X_val, y_train, y_val = train_test_split(Xy[:, :-1], Xy[:, -1], test_size=val_size)

    val_Xy_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        DATA_FOLDER,
        'val_Xy_save.npy'
    )
    np.save(val_Xy_save_file_name, np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1))

    return X_train, y_train