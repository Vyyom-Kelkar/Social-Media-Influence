import pandas as pd
import numpy as np
import sklearn as skl
import os

# This function is responsible for loading and processing the file into a
# pandas data frame
def read_to_dataframe(file_location):
    data = pd.read_csv(file_location, index_col=False).dropna()


    #Preprocess the data and/or delete extra columns
    #example ==> delete_extra_columns(data)

    return data

def load_data(file_name='./train.csv', reshape=False, cache=True, process=True):
    cache_save_location = file_name + '.pkl'

    if cache and os.path.exists(cache_save_location):
        print('load_data() - Loading cached version.')
        dataframe = pd.read_pickle(cache_save_location)
    else:
        print('load_data() - Computing dataframe.')
        dataframe = read_to_dataframe(file_name)
        if cache:
            print('load_data() - Caching dataframe.')
            dataframe.to_pickle(cache_save_location)

    if not process:
        return dataframe

    dataframe = skl.utils.shuffle(dataframe)
    print(dataframe)

load_data("../train.csv")
