"""
Script: model_training_pipeline.py
Description:
      This script runs the whole model training pipeline, from reading the training dataset,
      to printing the model scores on training and validation data, and the location of the saved model.
      It is the main script to run for training a new model on the raw dataset put in the data folder,
      and it calls all the other relevant modules in the src folder, in the right order.
      NB!: in addition to saved model there are saved preproc objects like the fitted encoders and scalers -
      these saves are also to be used by offline inferences reusing this model later on without retraining it.
"""

import os

import numpy as np
from loguru import logger

from src.eda_train_dataset import run_eda_train_dataset_f
from src.files_n_folders import DATA_FOLDER, MODELS_FOLDER, PROJECT_FOLDER
from src.ml_model_train_n_save import MODEL_SHORT_NAME, ml_model_train_n_save_f
from src.preproc_train_dataset import preproc_dataset_f
from src.read_train_dataset import read_train_dataset_f
from src.save_validation_data import save_validation_data_f

# run the model training pipeline
logger.info("reading raw dataset from file...")
ddf = read_train_dataset_f()
logger.debug("dataset loaded OK:\n{}".format(ddf.info()))

logger.info("running EDA on the dataset...")
num_flds, cat_flds, bin_flds = run_eda_train_dataset_f(ddf)
logger.debug(
    "EDA done OK, the dataset has {} numerical fields, {} categorical fields and {} binary fields.".format(
        len(num_flds), len(cat_flds), len(bin_flds)
    )
)

logger.info("preprocessing the dataset...")
Xy = preproc_dataset_f(ddf, num_flds, cat_flds, bin_flds)
logger.debug(
    "preprocessing done OK, the preprocessed dataset has this shape: {}".format(
        Xy.shape
    )
)

logger.info("saving the validation data for later use...")
X_train, y_train = save_validation_data_f(Xy)
logger.debug(
    "validation data saved OK, the training data has this shape: {}".format(
        X_train.shape
    )
)

logger.info("training the model and saving it to file...")
model = ml_model_train_n_save_f(X_train, y_train)
logger.debug("model trained and saved OK, the model is:\n{}".format(model))

logger.info("loading the validation data to check the model score on it...")
# load the validation data to be used to check the model score on it
Xy_val = np.load(os.path.join(PROJECT_FOLDER, DATA_FOLDER, "val_Xy_save.npy"))
X_val = Xy_val[:, :-1]
y_val = Xy_val[:, -1]

logger.info(
    "printing the model scores on training and validation data, and the location of the saved model..."
)
# print the model scores on training and validation data, and the location of the saved model
print("\nFYI:", MODEL_SHORT_NAME, "model has been trained OK:")
print("the model score on training data was :", model.score(X_train, y_train))
print("the model score on validation data is:", model.score(X_val, y_val))
print(
    "the trained model has been saved using joblib dump in this folder:",
    os.path.join(PROJECT_FOLDER, MODELS_FOLDER),
)
print("it was an MLOps-pleasure working with you!  See yah...\n")
