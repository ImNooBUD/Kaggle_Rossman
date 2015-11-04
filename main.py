__author__ = 'Dworkin'

from Settings import store_path, train_path, test_path, submission_path, store_descr_df_path, ext_sales_df_path, \
    rez_sparse_one_hot_path
from one_hot_encd import one_hot_dataframe
from locale import setlocale, LC_TIME
import pandas as pd
import numpy as np
from scipy import sparse
from datetime import datetime
import calendar
from dateutil.relativedelta import relativedelta
import pickle
import xgboost as xgb


def main():

    setlocale(LC_TIME, 'en_US.utf8')

    #store_df = pd.DataFrame.from_csv(store_path, sep=',', index_col=None)
    #train_df = pd.DataFrame.from_csv(train_path, sep=',', index_col=None)

    #descr_store_df = look_data_and_stats(store_df)
    #descr_store_df.to_csv(store_descr_df_path, sep=';')

    #ext_sales_df = extend_data(train_df, store_df)
    #ext_sales_df.to_csv(ext_sales_df_path, sep=';')

    ext_sales_df = pd.DataFrame.from_csv(ext_sales_df_path, sep=';')
    ext_sales_df = ext_sales_df[['Store', 'DayOfWeek', 'Sales', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                                 'StoreType', 'Assortment', 'CompetitionDistance', 'DateYear', 'DateMonth', 'DateDay',
                                 'DateWeekNumber', 'season', 'TimeDeltaCompetitionOpened', 'TimeDeltaPromoStart',
                                 'IsPromoInThisMonth', 'PreviousMonthSales', 'PreviousWeekSales']]
    ext_sales_df.loc[ext_sales_df.StateHoliday != '0', 'StateHoliday'] = '1'
    ext_sales_df.StateHoliday = ext_sales_df.StateHoliday.astype(np.int64)
    ext_sales_df.IsPromoInThisMonth = ext_sales_df.IsPromoInThisMonth.apply(lambda x: 1 if x else 0)

    #Long and not optimal way
    #ext_sales_df.Store = ext_sales_df.Store.astype(int).astype(str)
    #ext_sales_df, _, _ = one_hot_dataframe(ext_sales_df, ['Store', 'StoreType', 'Assortment', 'season'], replace=True)

    #ext_sales_df = pd.get_dummies(ext_sales_df, columns=['Store', 'StoreType', 'Assortment', 'season'], sparse=True)

    %%timeit
    one_hot_df = pd.get_dummies(ext_sales_df, columns=['StoreType', 'Assortment', 'season'], sparse=True)
    one_hot_df = one_hot_df.to_sparse(fill_value=0)
    #one_hot_df[one_hot_df.columns.tolist()].iloc[0:10000]

    #sparse_df_pickle_file = open(rez_sparse_one_hot_path, 'w')
    #pickle.dump(ext_sales_df, sparse_df_pickle_file, protocol=2)
    #sparse_df_pickle_file.close()

    sps_coo_matrix, rows, cols = sparse_df_to_saprse_matrix(one_hot_df)
    print sps_coo_matrix
    file = open('./Results/sps_coo_matrix', 'w')
    pickle.dump([sps_coo_matrix, rows, cols], file, protocol=2)
    file.close()

    """
    ext_sales_df = ext_sales_df.dropna()
    #dtrain = xgb.DMatrix(ext_sales_df[ext_sales_df.columns[~ext_sales_df.columns.isin(['Sales'])]].as_matrix(), ext_sales_df.Sales)
    param = {'bst:max_depth':20, 'bst:eta':1, 'objective':'reg:linear', 'subsample':1}
    param['nthread'] = 5
    plst = param.items()
    plst += [('eval_metric', 'rmse')]
    num_round = 39
    evallist  = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)


    sps_series_1 = sparse_df.Store_1
    sps_series_1.index = pd.MultiIndex.from_product([sparse_df.index.values.tolist(), ['Store_1']])
    A, rows, cols = sps_series_1.to_coo()

    sps_series_2 = sparse_df.Store_2
    sps_series_2.index = pd.MultiIndex.from_product([sparse_df.index.values.tolist(), ['Store_2']])
    B, rows2, cols2 = sps_series_2.to_coo()

    """

    return 0

def extend_data (sales_df, store_df):

    extended_sales_df = pd.merge(sales_df, store_df, on='Store')
    extended_sales_df['Date'] = extended_sales_df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

    extended_sales_df['DateYear'] = extended_sales_df.Date.apply(lambda x: x.year)
    extended_sales_df['DateMonth'] = extended_sales_df.Date.apply(lambda x: x.month)
    extended_sales_df['DateDay'] = extended_sales_df.Date.apply(lambda x: x.day)
    extended_sales_df['DateWeekNumber'] = extended_sales_df.Date.apply(lambda x: x.isocalendar()[1])
    extended_sales_df['season'] = extended_sales_df.DateMonth.apply(lambda x: get_season(x))

    extended_sales_df['DateConmpetitionOpened'] = \
        extended_sales_df.CompetitionOpenSinceYear.astype(str).apply(lambda x: x if x=='nan' else x[:-2]) \
        + '-' + extended_sales_df.CompetitionOpenSinceMonth.astype(str).apply(lambda x: x if x=='nan' else x[:-2]) + '-01'

    extended_sales_df['DateConmpetitionOpened'] = \
        extended_sales_df['DateConmpetitionOpened'].apply(lambda x: 'nan' if x.startswith('nan') else x)

    extended_sales_df.DateConmpetitionOpened = \
        extended_sales_df.DateConmpetitionOpened.apply(
            lambda x: datetime.strptime('2020-01-01', "%Y-%m-%d") if x == 'nan' else datetime.strptime(x, "%Y-%m-%d")
        )

    extended_sales_df['TimeDeltaCompetitionOpened'] = \
        (extended_sales_df.Date - extended_sales_df.DateConmpetitionOpened)/np.timedelta64(1, 'M')

    extended_sales_df.CompetitionDistance = extended_sales_df.CompetitionDistance.fillna(999999)


    extended_sales_df.Promo2SinceYear = extended_sales_df.Promo2SinceYear.fillna(2020.0)
    extended_sales_df.Promo2SinceWeek = extended_sales_df.Promo2SinceWeek.fillna(1.0)
    extended_sales_df['DatePromo2Start'] = \
        extended_sales_df.Promo2SinceYear.astype(int).astype(str) + \
        '-' + extended_sales_df.Promo2SinceWeek.astype(int).astype(str) + '-1'
    extended_sales_df.DatePromo2Start = \
        extended_sales_df.DatePromo2Start.apply(lambda x: datetime.strptime(x, "%Y-%W-%w"))

    extended_sales_df['TimeDeltaPromoStart'] = \
        (extended_sales_df.Date - extended_sales_df.DatePromo2Start)/np.timedelta64(1, 'M')


    extended_sales_df['IsPromoInThisMonth'] = pd.Series(np.nan, extended_sales_df.index)
    extended_sales_df.set_value(extended_sales_df.TimeDeltaPromoStart < 0, 'IsPromoInThisMonth', False)

    extended_sales_df['DateMonthAbbr'] = extended_sales_df.DateMonth.apply(lambda x: calendar.month_abbr[x])

    extended_sales_df['IsPromoInThisMonth'] = [x in y for x, y in zip(extended_sales_df['DateMonthAbbr'].astype(str),
                                                        extended_sales_df['PromoInterval'].astype(str))]

    group_sales_monthly = extended_sales_df.groupby(['Store', 'DateYear', 'DateMonth'])['Sales'].sum()

    extended_sales_df['PreviousMonthSales'] = [
        group_sales_monthly[x, (y-relativedelta(months=1)).year, (y-relativedelta(months=1)).month]
        if (x, (y-relativedelta(months=1)).year, (y-relativedelta(months=1)).month) in group_sales_monthly.index else np.nan
        for x, y in zip(extended_sales_df.Store, extended_sales_df.Date)
        ]


    group_sales_weekly = extended_sales_df.groupby(['Store', 'DateYear', 'DateWeekNumber'])['Sales'].sum()

    extended_sales_df['PreviousWeekSales'] = [
        group_sales_weekly[x, (y-relativedelta(weeks=1)).year, (y-relativedelta(weeks=1)).isocalendar()[1]]
        if (x, (y-relativedelta(weeks=1)).year, (y-relativedelta(weeks=1)).isocalendar()[1]) in group_sales_weekly.index else np.nan
        for x, y in zip(extended_sales_df.Store, extended_sales_df.Date)
        ]




    return extended_sales_df


def get_season (month):

    season = None
    if month in (12, 1, 2):
        season = 'winter'
    elif month in (3, 4, 5):
        season = 'spring'
    elif month in (6, 7, 8):
        season = 'summer'
    elif month in (9, 10, 11):
        season = 'autumn'

    return season

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

def sparse_df_to_saprse_matrix (sparse_df):

    index_list = sparse_df.index.values.tolist()
    matrix_columns = []
    sparse_matrix = None

    for column in sparse_df.columns:
        sps_series = sparse_df[column]
        sps_series.index = pd.MultiIndex.from_product([index_list, [column]])
        curr_sps_column, rows, cols = sps_series.to_coo()
        if sparse_matrix != None:
            sparse_matrix = sparse.hstack([sparse_matrix, curr_sps_column])
        else:
            sparse_matrix = curr_sps_column
        matrix_columns.extend(cols)

    return sparse_matrix, index_list, matrix_columns

if __name__ == '__main__':
    main()

