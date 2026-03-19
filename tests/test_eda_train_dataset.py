'''
Module: test_eda_train_dataset.py
Description:
    Unit tests for the function run_eda_train_dataset_f in src/eda_train_dataset.py
    The tests will check if the function correctly identifies the types of fields
'''

from src.eda_train_dataset import run_eda_train_dataset_f
from src.read_train_dataset import read_train_dataset_f

def test_run_eda_train_dataset_f():
    df = read_train_dataset_f()
    eda_results = run_eda_train_dataset_f(df)

    assert type(eda_results) is tuple
    assert len(eda_results) == 3
    assert len(eda_results[0]) + len(eda_results[1]) + len(eda_results[2]) == len(df.columns)
    assert run_eda_train_dataset_f(df,2) == 'ERROR IN PARAMS VALUE: when calling run_eda_train_dataset_f, cat_thres must be greater than 2'
    