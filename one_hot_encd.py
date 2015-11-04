# -*- coding: utf-8 -*-
""" Small script that shows hot to do one hot encoding
    of categorical columns in a pandas DataFrame.
    See:
    http://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
    http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.DictVectorizer.html
"""
import pandas
import random
import numpy
from sklearn.feature_extraction import DictVectorizer


def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    vecData = pandas.DataFrame(vec.fit_transform(data[cols].to_dict(orient='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)


def main():

    # Get a random DataFrame
    df = pandas.DataFrame(numpy.random.randn(25, 3), columns=['a', 'b', 'c'])

    # Make some random categorical columns
    df['e'] = [random.choice(('Chicago', 'Boston', 'New York')) for i in range(df.shape[0])]
    df['f'] = [random.choice(('Chrome', 'Firefox', 'Opera', "Safari")) for i in range(df.shape[0])]
    print df

    # Vectorize the categorical columns: e & f
    df, _, _ = one_hot_dataframe(df, ['e', 'f'], replace=True)
    print df

if __name__ == '__main__':
    main()
