__author__ = 'Dworkin'

from Settings import store_path, train_path, test_path, submission_path, store_descr_df_path
from one_hot_encd import one_hot_dataframe

import pandas as pd
import numpy as np
def main():

    store_df = pd.DataFrame.from_csv(store_path, sep=',')
    #train_df = pd.DataFrame.from_csv(train_path, sep=',')

    descr_store_df = look_data_and_stats(store_df)
    descr_store_df.to_csv(store_descr_df_path, sep=';')

    return 0


def look_data_and_stats (input_df, cols=None):

    if cols == None:
        cols = input_df.columns

    description_df = pd.DataFrame(columns=cols)
    for col in cols:

        # Debug info
        # print col


        total_len = len(input_df[col])

        #Column type
        description_df.set_value('col_type', col, input_df[col].dtype)

        #Contain NAN
        contain_nan = any(input_df[col].isnull())
        description_df.set_value('contain_nan', col, contain_nan)

        #volume of NAN
        volume_nan  = float(100*sum(input_df[col].isnull()))/total_len
        description_df.set_value('volume_nan_percent', col, volume_nan)

        #Number of unique values
        number_of_uniques = len(input_df[col].unique())
        description_df.set_value('number_of_uniques', col, number_of_uniques)
        #volume of unique values
        description_df.set_value('unique_to_all_records_percent', col, float(100*number_of_uniques)/total_len)


        #create distribution of DF
        volume_df = pd.DataFrame(columns=['volume'])

        for value in input_df[col].unique():
            # Debug info
            # print value

            if str(value) == 'nan':
                volume = float(100*len(input_df[input_df[col].isnull() == True]))/total_len
                volume_df.set_value('nan', 'volume', volume)
            else:
                volume = float(100*len(input_df[input_df[col] == value][col]))/total_len
                volume_df.set_value(value, 'volume', volume)


        #sum volume of unique with less then 5% of total volume
        total_volume_of_rare_cases = volume_df[volume_df['volume'] < 5.0]['volume'].sum()
        description_df.set_value('total_volume_of_rare_cases', col, total_volume_of_rare_cases)

        #Top 10 unique values and sum volume of top 10 unique values
        top_10_valuse = volume_df.sort(['volume'], ascending=False)['volume'].head(10).index.values
        volume_top_10 = volume_df.ix[top_10_valuse]['volume'].sum()
        description_df.set_value('volume_top_10', col, volume_top_10)
        description_df.set_value('top_10_values', col, top_10_valuse)

    return description_df

if __name__ == '__main__':
    main()
