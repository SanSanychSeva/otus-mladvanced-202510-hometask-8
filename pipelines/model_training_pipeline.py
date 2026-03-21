import os
import numpy as np

from src.files_n_folders import PROJECT_FOLDER, DATA_FOLDER, MODELS_FOLDER
from src.preproc_train_dataset import preproc_dataset_f
from src.save_validation_data import save_validation_data_f
from src.read_train_dataset import read_train_dataset_f
from src.eda_train_dataset import run_eda_train_dataset_f
from src.ml_model_train_n_save import ml_model_train_n_save_f, MODEL_SHORT_NAME

ddf                          = read_train_dataset_f()
num_flds, cat_flds, bin_flds = run_eda_train_dataset_f(ddf)
Xy                           = preproc_dataset_f(ddf, num_flds, cat_flds, bin_flds)
X_train, y_train             = save_validation_data_f(Xy)
model                        = ml_model_train_n_save_f(X_train, y_train)

# load the validation data and check the model score on it
Xy_val = np.load(os.path.join(PROJECT_FOLDER, DATA_FOLDER, 'val_Xy_save.npy'))
X_val = Xy_val[:,:-1]
y_val = Xy_val[:,-1]

# print the model scores on training and validation data, and the location of the saved model
print('\nFYI:', MODEL_SHORT_NAME, 'model has been trained OK:')
print('the model score on training data was :', model.score(X_train, y_train))
print('the model score on validation data is:', model.score(X_val, y_val))
print('the trained model has been saved using joblib dump in this folder:', 
      os.path.join(PROJECT_FOLDER, MODELS_FOLDER))
print('it was an MLOps-pleasure working with you!  See yah...\n')