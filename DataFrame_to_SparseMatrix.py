__author__ = 'dworkin'

import pandas as pd
import scipy.sparse as sps
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
import itertools


def get_feature_unique_values(train_df_path, test_df_path=None, cols=None):
    """
    Due to memmory problems
    :param train_df_path: path to train data
    :param test_df_path: path to test data
    :param cols: cols for 1-hot-encoding (if not set, will be used all from train)
    :return: list of dicts [{col1:[value1, value2, ...]}, {col2:[...]}, ...]
    """

    rez_dicts = {}
    file = open(train_df_path, 'r')
    train_df = pd.DataFrame(pickle.load(file))
    file.close()

    if cols == None:
        cols = train_df.columns

    if set(cols).issubset(train_df.columns) == False:
        print "Error in get_feature_unique_values: {0} NOT a subset of dataframe columns".format(set(cols))
        return {}

    for curr_col in cols:
        rez_dicts[curr_col] = train_df[curr_col].unique().tolist()

    del train_df

    if test_df_path:
        file = open(test_df_path, 'r')
        test_df = pd.DataFrame(pickle.load(file))
        file.close()

        test_cols = test_df.columns
        test_keys = set(test_cols).intersection(rez_dicts.keys())

        for key in test_keys:
            rez_dicts[key].extend(test_df[key].unique().tolist())
            rez_dicts[key] = list(set(rez_dicts[key]))


    return rez_dicts

def get_dict_vectorizers(dict_of_key_values):
    """
    FEATURES MUST HAVE STRING TYPE VALUES AS IN DATAFRAME (otherwise they wouldn't be encoded)
    :param dict_of_key_values:
    :return:
    """

    #TODO find optimal way
    all_pairs_dict = []
    generators = [itertools.product([x], dict_of_key_values[x]) for x in dict_of_key_values.keys()]
    for gen in generators:
        all_pairs_dict.extend(
            [dict([curr_pair]) for curr_pair in gen]
        )

    feature_vect = DictVectorizer(sparse=True, separator='_')
    feature_vect.fit(all_pairs_dict)

    #print all_pairs_dict
    # pd.DataFrame.to_dict(orient='records')

    return feature_vect