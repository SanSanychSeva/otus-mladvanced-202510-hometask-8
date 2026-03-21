import os
from joblib import dump as save_fld_handler

from sklearn.ensemble import GradientBoostingClassifier

from src.files_n_folders import PROJECT_FOLDER, MODELS_FOLDER

MODEL_SHORT_NAME = 'GBC'

def ml_model_train_n_save_f(X_train, y_train):
                     # select model here
    model = GradientBoostingClassifier()     # select model here
    model.fit(X_train, y_train)

    model_save_file_name = os.path.join(
        PROJECT_FOLDER, 
        MODELS_FOLDER, 
        'model_save_' + MODEL_SHORT_NAME + '.bin'
    )

    save_fld_handler(model, model_save_file_name, compress=True)
    return model