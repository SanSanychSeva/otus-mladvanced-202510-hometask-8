import os
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier

from src.files_n_folders import PROJECT_FOLDER, MODELS_FOLDER
from src.preproc_train_dataset import save_fld_handler

def ml_model_train_n_save_f(X_train, y_train):
    model_short_name = 'GBC'                 # select model here
    model = GradientBoostingClassifier()     # select model here
    model.fit(X_train, y_train)

    model_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER, 
        'model_save_' + model_short_name + '-' + str(datetime.now()).split(' ')[0] + '.bin'
    )

    save_fld_handler(model, model_save_file_name, compress=True)
    return model