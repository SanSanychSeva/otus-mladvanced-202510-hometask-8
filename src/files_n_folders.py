'''
Module: files_n_folders.py
Description:
    the module assembles all named constants reflecting the home folder structure
'''

PROJECT_FOLDER = '..'          # all folders are supposed to be inside project - thus, just go up!

DATA_FOLDER = 'data'           # inside project
DATASET_NAME = 'heart.csv'     # inside data 
                               # NB!: this is the source data-file with model training dataset

MODELS_FOLDER = 'models'       # inside project
CONFIG_FOLDER = 'config'       # inside project
PREPROC_FOLDER = 'preproc'     # inside models 
                               # NB!: in addition to model we save all trained preproc objects 