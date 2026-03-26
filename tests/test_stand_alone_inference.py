'''
Module: test_stand_alone_inference.py
Description: 
    This module contains tests for the stand-alone inference functionality of the model. 
    It verifies that the predictions made by the model on the training dataset are accurate, 
    ensuring that the model is correctly loaded and functioning as expected.
'''
from sklearn.metrics import accuracy_score

from src.read_train_dataset import read_train_dataset_f
from src.stand_alone_inference import load_model_n_predict

def test_stand_alone_inference_f():
    ddf = read_train_dataset_f()
    new_data = ddf.drop(columns='target')

    predictions = load_model_n_predict(new_data)

    assert accuracy_score(ddf['target'], predictions) > 0.98