import numpy as np
import os

from src.files_n_folders import PROJECT_FOLDER, DATA_FOLDER
from src.preproc_train_dataset import preproc_dataset_f
from src.save_validation_data import keep_validation_data_f
from src.read_train_dataset import read_train_dataset_f
from src.eda_train_dataset import run_eda_train_dataset_f

def test_save_validation_data_f():
    ddf = read_train_dataset_f()
    num_flds, cat_flds, num_cat_flds = run_eda_train_dataset_f(ddf)
    Xy = preproc_dataset_f(ddf, num_flds, cat_flds, num_cat_flds)

    X_train, y_train = keep_validation_data_f(Xy)

    val_Xy_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        DATA_FOLDER,
        'val_Xy_save.npy'
    )
    assert os.path.exists(val_Xy_save_file_name)
    assert len(X_train.shape) == 2 and len(y_train.shape) == 1