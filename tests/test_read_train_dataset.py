'''
Module: test_read_train_dataset.py
Description:
    This module contains tests for the read_train_dataset_f function
'''
import os

from src.read_train_dataset import read_train_dataset_f

def test_read_train_dataset_f():
    df = read_train_dataset_f()
    assert df is not None
    assert df.shape == (1025, 14)