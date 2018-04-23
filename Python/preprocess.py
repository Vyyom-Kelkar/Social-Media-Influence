import pandas as pd
import numpy as np
import sklearn as skl
import os

# This function is responsible for loading and processing the file into a
# pandas data frame
def read_to_dataframe(file_location):
    data = pd.read_csv(file_location, index_col=False).dropna()


    #Preprocess the data and/or delete extra columns
    delete_extra_columns(data)

    return data

def delete_extra_columns(data):
    cols_to_delete = ['A_listed_count', 'A_mentions_sent', 'A_following_count', 'A_retweets_sent',
            'A_network_feature_1', 'A_network_feature_2', 'A_network_feature_3','B_mentions_sent',
            'B_listed_count', 'B_following_count', 'B_retweets_sent','B_network_feature_1',
            'B_network_feature_2','B_network_feature_3', 'A_follower_count', 'B_follower_count']

    for col in cols_to_delete:
        del data[col]

def load_data(file_name='train.csv', reshape=False, cache=True, process=True):
    cache_save_location = file_name + '.pkl'
    #
    # if cache and os.path.exists(cache_save_location):
    #     print('load_data() - Loading cached version.')
    #     dataframe = pd.read_pickle(cache_save_location)
    # else:
    print('load_data() - Computing dataframe.')
    dataframe = read_to_dataframe(file_name)
    if cache:
        print('load_data() - Caching dataframe.')
        dataframe.to_pickle(cache_save_location)

    if not process:
        return dataframe

    dataframe = skl.utils.shuffle(dataframe)
    A_score = dataframe['A_follower_count']*dataframe['A_mentions_received']*dataframe['A_retweets_received']*dataframe['A_posts']
    B_score = dataframe['B_follower_count']*dataframe['B_mentions_received']*dataframe['B_retweets_received']*dataframe['B_posts']
    my_choice = []
    correct=0
    incorrect=0

    for i in range(len(A_score)):
        if A_score[i] > B_score[i] and dataframe['Choice'][i] ==1:
            correct+=1
        else:
            incorrect+=1

    correct=float(correct)
    total = int(correct+incorrect);
    percentage = float(correct/total)*100
    print(percentage)

load_data("../train.csv")
