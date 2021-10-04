import pandas as pd


def generate_features(train: pd.DataFrame) -> pd.DataFrame:
    # last u_in
    train['last_value_u_in'] = train.groupby('breath_id')['u_in'].transform('last')

    # create lag in data
    step_num = round(train.shape[0]/(5*train['breath_id'].nunique()))
    print(step_num)
    for lag in range(-step_num, step_num, 1):
        train[f'u_in_lag_{str(lag)}'] = train.groupby('breath_id')['u_in'].shift(lag)
        train[f'u_out_lag_{str(lag)}'] = train.groupby('breath_id')['u_out'].shift(lag)

    # first u_in, u_out
    train['u_in_first'] = train.groupby('breath_id')['u_in'].first()
    train['u_out_first'] = train.groupby('breath_id')['u_out'].first()

    # min max median mean
    for func in ['min', 'max', 'median', 'mean']:
        train[f'breath_id__u_in__{func}'] = train.groupby(['breath_id'])['u_in'].transform(func)
        train[f'breath_id__u_out__{func}'] = train.groupby(['breath_id'])['u_out'].transform(func)

    # difference between consecutive (последовательными) values
    for lag in ['1', '2', '3', '4']:
        train[f'u_in_diff_{lag}'] = train['u_in'] - train[f'u_in_lag_{lag}']
        train[f'u_out_diff_{lag}'] = train['u_out'] - train[f'u_out_lag_{lag}']
    # from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
    train.loc[train['time_step'] == 0, 'u_in_diff'] = 0
    train.loc[train['time_step'] == 0, 'u_out_diff'] = 0

    # difference between the current value of u_in and the min, max, median values within the breath
    train['breath_id__u_in__diffmin'] = train.groupby(['breath_id'])['u_in'].transform('min') - train['u_in']
    train['breath_id__u_in__diffmax'] = train.groupby(['breath_id'])['u_in'].transform('max') - train['u_in']
    train['breath_id__u_in__diffmean'] = train.groupby(['breath_id'])['u_in'].transform('mean') - train['u_in']

    # OHE
    # train['R_div_C'] = train['R'].div(train['C']) ухудшило
    train['R__C'] = train['R'].astype(str) + '__' + train['C'].astype(str)
    train = train.merge(pd.get_dummies(train['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
    train = train.merge(pd.get_dummies(train['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
    train = train.merge(pd.get_dummies(train['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'],
                                                                                                              axis=1)
    # CUMSUM https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/273974
    train['u_in_cumsum'] = train.groupby(['breath_id'])['u_in'].cumsum()
    train['time_step_cumsum'] = train.groupby(['breath_id'])['time_step'].cumsum()

    # очень маленькая прибавка, возможно и не надо?
    # time since last step
    train['time_step_diff'] = train.groupby('breath_id')['time_step'].diff().fillna(0)
    # rolling window ts feats
    train['ewm_u_in_mean'] = train.groupby('breath_id')['u_in'].ewm(halflife=9).mean().reset_index(level=0, drop=True)
    train['ewm_u_in_std'] = train.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0, drop=True)
    train['ewm_u_in_corr'] = train.groupby('breath_id')['u_in'].ewm(halflife=15).corr().reset_index(level=0, drop=True)
    # rolling window of 15 periods
    train[['15_in_sum', '15_in_min', '15_in_max', '15_in_mean', '15_out_std']] = train.groupby('breath_id')[
        'u_in'].rolling(window=15, min_periods=1).agg(
        {'15_in_sum': 'sum', '15_in_min': 'min', '15_in_max': 'max', '15_in_mean': 'mean',
         '15_in_std': 'std'}).reset_index(level=0, drop=True)
    train[['45_in_sum', '45_in_min', '45_in_max', '45_in_mean', '45_out_std']] = train.groupby('breath_id')[
        'u_in'].rolling(window=45, min_periods=1).agg(
        {'45_in_sum': 'sum', '45_in_min': 'min', '45_in_max': 'max', '45_in_mean': 'mean',
         '45_in_std': 'std'}).reset_index(level=0, drop=True)

    train[['15_out_mean']] = train.groupby('breath_id')['u_out'].rolling(window=15, min_periods=1).agg(
        {'15_out_mean': 'mean'}).reset_index(level=0, drop=True)

    # feature by u in or out (ideally - make 2 sep columns for each state) # dan
    train['u_in_partition_out_sum'] = train.groupby(['breath_id', 'u_out'])['u_in'].transform('sum')

    train = train.fillna(0)  # дополнить к данным со смещениями 0 значения
    return train
