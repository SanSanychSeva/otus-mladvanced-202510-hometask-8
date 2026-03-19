'''
Module: files_n_folders.py
Description:
    the module assembles all named constants reflecting the home folder structure
'''

PROJECT_FOLDER = '.'          # all folders are supposed to be inside project

DATA_FOLDER = 'data'           # inside project
DATASET_NAME = 'heart.csv'     # inside data 
                               # NB!: this is the source data-file with model training dataset

MODELS_FOLDER = 'models'       # inside project
CONFIG_FOLDER = 'config'       # inside project
PREPROC_FOLDER = 'preproc'     # inside models 
                               # NB!: in addition to model we save all trained preproc objects 

def files_n_folders_f() -> dict[str, str]:
    '''
    the function returns all named constants reflecting the home folder structure
    '''
    return {
        'PROJECT_FOLDER': PROJECT_FOLDER,
        'DATA_FOLDER': DATA_FOLDER,
        'DATASET_NAME': DATASET_NAME,
        'MODELS_FOLDER': MODELS_FOLDER,
        'CONFIG_FOLDER': CONFIG_FOLDER,
        'PREPROC_FOLDER': PREPROC_FOLDER
    }