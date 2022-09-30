import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def preprocessing(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df[['cc_num']] = df[['cc_num']].astype('object')
    df['name'] = df['first'] + ' ' + df['last']
    df['age'] = np.round((df['trans_date_trans_time'] - df['dob'])/np.timedelta64(1, 'Y'))
    df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
    df['day'] = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek
    df['month'] = pd.to_datetime(df['trans_date_trans_time']).dt.month
    df.drop(['first', 'last', 'trans_num', 'dob'], axis=1, inplace = True)
    df.sort_values(by='trans_date_trans_time', inplace=True)

    return df


def feature_selection(df):
    features = ['category', 'amt', 'city_pop', 'merch_lat', 'merch_long',
    'age', 'hour', 'day', 'month', 'is_fraud']

    df = pd.get_dummies(df[features], drop_first=True)

    y = df['is_fraud'].values
    X = df.drop('is_fraud', axis='columns').values

    return X, y


def data_balancing(X_train, y_train):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    return X_resampled, y_resampled
