'''
Module: eda_train_dataset.py
Description: 
    module for exploratory data analysis of the training dataset, to be used before the pre-processing
'''

import pandas

def run_eda_train_dataset_f(df:'pandas.DataFrame',
                            cat_thres:int=9          # discriminates categorical fields (tuning)
                            ) -> (list, list, list):
    '''
    explore fields values and returns lists of numerical, categorical and binary fields
    '''
    if cat_thres <= 2:
        return 'ERROR IN PARAMS VALUE: when calling run_eda_train_dataset_f, cat_thres must be greater than 2'
        
    flds_list = df.columns
    num_flds = []
    cat_flds = []
    bin_flds = []

    for i in range(len(flds_list)):
        fld = flds_list[i]
        view = df[fld].value_counts()
       
        if len(view) > cat_thres:
            num_flds.append(fld)
        elif len(view) == 2:
            bin_flds.append(fld)
        else:
            cat_flds.append(fld)
    
    return num_flds, cat_flds, bin_flds