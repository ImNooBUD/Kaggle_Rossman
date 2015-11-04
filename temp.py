__author__ = 'Dworkin'

import Settings
import scipy.sparse as sp
import pandas as pd
import numpy as np
import datetime
import pickle
import io
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from fastFM import mcmc
from sklearn.metrics import mean_squared_error

import xgboost as xgb


def dirty_trick():
    users = pd.DataFrame.from_csv(data_dir_path+Settings.user_list, index_col=False)
    null_users = users[(users.WITHDRAW_DATE.isnull()==False)&(users.WITHDRAW_DATE<'2012-06-24')]['USER_ID_hash']
    sample_submission_df = pd.DataFrame.from_csv(data_dir_path+Settings.sample_submission, index_col=False)

    sample_submission_df.ix[sample_submission_df.USER_ID_hash.isin(null_users)==False,
                            'PURCHASED_COUPONS'] = 'c9e1dcbd8c98f919bf85ab5f2ea30a9d 2fcca928b8b3e9ead0f2cecffeea50c1 0fd38be174187a3de72015ce9d5ca3a2'


    sample_submission_df.to_csv('Lucky_coupon_without_withdraw.csv', index=False)


def get_feature_vectorizer (df, cols):
    """

    :return: vectorizers 1-hot-encoding for feature
    """

    feature_vect = DictVectorizer(sparse=True)
    feature_vect.fit_transform(df[cols].to_dict(outtype='records'))
    return feature_vect

def split_dataframe (df, split_step):
    """

    :param df:
    :param split_step:
    :return:
    """
    row_count = len(df.index)
    array_of_df = []

    if row_count <= split_step:
        array_of_df.append(df)
    else:
        start_indx = 0
        while start_indx < row_count:
            end_indx = start_indx + split_step if (start_indx+split_step) <= row_count else row_count
            array_of_df.append(df.iloc[start_indx:end_indx, :])
            start_indx += split_step

    return array_of_df

#TODO change - input valuse must also include vectorizers
def prepare_train_matrix(users_df, pref_df, coupon_desc_df, view_log_df):
    """

    :param users_df:
    :param pref_df:
    :param coupon_desc_df:
    :param view_log_df:
    :return:
    """
    user_hash_vect = get_feature_vectorizer(users_df, ['USER_ID_hash'])
    sex_vect = get_feature_vectorizer(users_df, ['SEX_ID'])
    prefect_vect = get_feature_vectorizer(pref_df, ['PREF_NAME'])
    capsule_text_vect = get_feature_vectorizer(coupon_desc_df, ['CAPSULE_TEXT'])

    array_of_view_log_df = split_dataframe(view_log_df, 100000)
    rez_coo_matrix = None
    y_train = None

    for part_df in array_of_view_log_df:
        merge_view_and_coupons = pd.merge(part_df, coupon_desc_df, how='left',
                                          left_on='VIEW_COUPON_ID_hash', right_on='COUPON_ID_hash')
        merge_total = pd.merge(merge_view_and_coupons, users_df, how='left', on='USER_ID_hash')

        #think about how to deal with NaN values
        merge_total = merge_total[pd.notnull(merge_total['PRICE_RATE']) & pd.notnull(merge_total['DISCOUNT_PRICE'])]

        users_hash = [{'USER_ID_hash': x} for x in merge_total['USER_ID_hash']]
        users_sex = [{'SEX_ID': x} for x in merge_total['SEX_ID']]
        users_age = [[x] for x in merge_total['AGE']]
        users_pref = [{'PREF_NAME': x} for x in merge_total['PREF_NAME']]

        coupon_capsule_text = [{'CAPSULE_TEXT': x} for x in merge_total['CAPSULE_TEXT']]
        coupon_price_rate = [[x] for x in merge_total['PRICE_RATE']]
        coupon_discount_price = [[x] for x in merge_total['DISCOUNT_PRICE']]
        coupon_pref = [{'PREF_NAME': x} for x in merge_total['ken_name']]

        coupon_purchase = [x for x in merge_total['PURCHASE_FLG']]

        temp_coo_matrix = sp.hstack([
                user_hash_vect.transform(users_hash),
                sex_vect.transform(users_sex),
                prefect_vect.transform(users_pref),
                users_age,
                capsule_text_vect.transform(coupon_capsule_text),
                coupon_price_rate,
                coupon_discount_price,
                prefect_vect.transform(coupon_pref)
            ])

        if rez_coo_matrix is None:
            rez_coo_matrix = temp_coo_matrix
        else:
            rez_coo_matrix = sp.vstack([rez_coo_matrix, temp_coo_matrix])

        if y_train is None:
            y_train = coupon_purchase
        else:
            y_train.extend(coupon_purchase)


    return rez_coo_matrix, y_train

def prepare_view_log_df (view_log):

    view_log = pd.DataFrame(view_log)
    #TODO make through normal datetime.to_date()
    view_log['I_DATE'] = view_log['I_DATE'].map(lambda x: x[0:10])
    view_log = view_log.drop_duplicates(['PURCHASE_FLG', 'I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash'])
    return view_log

#TODO change - input valuse must also include vectorizers
def get_predicted_coupons(users_df, pref_df, coupon_desc_df, classifier, X_train, Y_train, test_coupons_df, split_step = 25000):

    user_hash_vect = get_feature_vectorizer(users_df, ['USER_ID_hash'])
    sex_vect = get_feature_vectorizer(users_df, ['SEX_ID'])
    prefect_vect = get_feature_vectorizer(pref_df, ['PREF_NAME'])
    capsule_text_vect = get_feature_vectorizer(coupon_desc_df, ['CAPSULE_TEXT'])

    users_df = users_df.set_index([[0]*len(users_df)])
    test_coupons_df = test_coupons_df.set_index([[0]*len(test_coupons_df)])
    df_cartesian = users_df.join(test_coupons_df, how='outer')

    array_of_df_cartesian = split_dataframe(df_cartesian, split_step)
    rez_test_coo_matrix = None

    for partition_df in array_of_df_cartesian:
        users_hash = [{'USER_ID_hash': x} for x in partition_df['USER_ID_hash']]
        users_sex = [{'SEX_ID': x} for x in partition_df['SEX_ID']]
        users_age = [[x] for x in partition_df['AGE']]
        users_pref = [{'PREF_NAME': x} for x in partition_df['PREF_NAME']]

        coupon_capsule_text = [{'CAPSULE_TEXT': x} for x in partition_df['CAPSULE_TEXT']]
        coupon_price_rate = [[x] for x in partition_df['PRICE_RATE']]
        coupon_discount_price = [[x] for x in partition_df['DISCOUNT_PRICE']]
        coupon_pref = [{'PREF_NAME': x} for x in partition_df['ken_name']]

        temp_coo_matrix = sp.hstack([
                    user_hash_vect.transform(users_hash),
                    sex_vect.transform(users_sex),
                    prefect_vect.transform(users_pref),
                    users_age,
                    capsule_text_vect.transform(coupon_capsule_text),
                    coupon_price_rate,
                    coupon_discount_price,
                    prefect_vect.transform(coupon_pref)
                ])

        if rez_test_coo_matrix is None:
            rez_test_coo_matrix = temp_coo_matrix
        else:
            rez_test_coo_matrix = sp.vstack([rez_test_coo_matrix, temp_coo_matrix])

    purch_chance = classifier.fit_predict_proba(rez_coo_matrix.tocsc(), y_train, rez_test_coo_matrix.tocsc())
    np.save(data_dir_path+'predicted_purch_chance.npy', purch_chance)

    df_cartesian.loc[:, 'purch_chance'] = purch_chance

    return df_cartesian


if __name__ == '__main__':

    #dirty_trick()

    #train data
    """
    users = pd.DataFrame.from_csv(data_dir_path+Settings.user_list, index_col=False)
    coupons = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_list_train, index_col=False)
    view_log = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_visit_train, index_col=False)
    view_log = prepare_view_log_df(view_log)
    purchase_log = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_detail_train, index_col=False)
    coupon_area = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_area_train, index_col=False)
    prefect = pd.DataFrame.from_csv(data_dir_path+Settings.prefecture_locations, index_col=False)
    prefect.columns = ['PREF_NAME', 'PREFECTUAL_OFFICE', 'LATITUDE', 'LONGITUDE']
    #test data
    coupons_test = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_list_test, index_col=False)
    """

    """
    rez_coo_matrix, y_train = \
        prepare_train_matrix(users_df=users, pref_df=prefect, coupon_desc_df=coupons, view_log_df=view_log)

    print rez_coo_matrix.shape
    print 'Matrix DONE! Trying to save'

    np.save('../total_matrix_cols.npy', rez_coo_matrix.col)
    np.save('../total_matrix_data.npy', rez_coo_matrix.data)
    np.save('../total_matrix_row.npy', rez_coo_matrix.row)
    np.save('../total_matrix_shape.npy', rez_coo_matrix.shape)
    np.save('../y_train.npy', np.asarray(y_train))
    print 'DONE'
    """
    """
    cols = np.load('../total_matrix_cols.npy')
    data = np.load('../total_matrix_data.npy')
    rows = np.load('../total_matrix_row.npy')
    shape = np.load('../total_matrix_shape.npy')
    rez_coo_matrix = sp.coo_matrix((data, (rows, cols)), shape)

    y_train = np.asarray(np.load('../y_train.npy'))

    fm_clf = mcmc.FMClassification()

    rez = get_predicted_coupons(users_df=users, pref_df=prefect, coupon_desc_df=coupons, classifier=fm_clf,
                                X_train=rez_coo_matrix, Y_train=y_train, test_coupons_df=coupons_test)

    rez.to_csv(data_dir_path+'rez_prediction.csv', index=False)
    """
    rez = pd.DataFrame.from_csv(data_dir_path+'rez_prediction.csv')

    top_coupons = pd.DataFrame(columns=['USER_ID_hash', 'PURCHASED_COUPONS'])

    #TODO delete after opening comments
    purchase_log = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_detail_train, index_col=False)

    #TODO exclude both earlie coupons
    for user in rez['USER_ID_hash'].unique():
        temp_dict = {'USER_ID_hash': user, 'PURCHASED_COUPONS': None}
        sorted_user_prob_df = rez[rez['USER_ID_hash'] == user][['COUPON_ID_hash', 'purch_chance']].sort('purch_chance', ascending=False)

        test_coupons = sorted_user_prob_df['COUPON_ID_hash'].unique()[0:25]
        earlier_buyed_coupons = purchase_log[purchase_log['USER_ID_hash'] == user]['COUPON_ID_hash'].unique()

        top_10_one_string = ''
        i = 0
        for coupon_hash in test_coupons:
            if coupon_hash not in earlier_buyed_coupons:
                top_10_one_string += coupon_hash + ' '
                i += 1
            else:
                print 'bayed earlier user {0} - coupon {1}'.format(user, coupon_hash)

            if i >=10: break

        temp_dict['PURCHASED_COUPONS'] = top_10_one_string
        top_coupons = top_coupons.append(temp_dict, ignore_index=True)

    top_coupons.to_csv(data_dir_path+'predicted_top_10_purchases.csv', index=False)
    print 'FINISH HIM!!!'

    """
    withdraw_purchase = view_log[
        (view_log.USER_ID_hash.isin(users[users.WITHDRAW_DATE.isnull()==False].USER_ID_hash))
        & (view_log.PURCHASE_FLG==1)]
    last_purchase_withdraw = withdraw_purchase.groupby(['USER_ID_hash'],sort=False, as_index=False)['I_DATE'].max()

    updated_withdraw_users = pd.merge(users, last_purchase_withdraw, on='USER_ID_hash')
    """
    pass



