import pandas as pd


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    # данные нужны где uo=0
    # max_len = df.drop(df[df.u_out != 0].index).breath_id.value_counts().max()  # 32
    # min_len = df.drop(df[df.u_out != 0].index).breath_id.value_counts().min()  # 25
    # print(max_len)
    # print(min_len)

    # last u_in
    df['last_value_u_in'] = df.groupby('breath_id')['u_in'].transform('last')

    # increase resistance
    df['t_increase'] = df['R']*df['C']

    # create lag in data (можно 5 убрать)
    df.rename(columns={'time_step': 'time_step_lag_0'}, inplace=True)
    df.rename(columns={'u_in': 'u_in_lag_0'}, inplace=True)
    df.rename(columns={'u_out': 'u_out_lag_0'}, inplace=True)

    steps_num = 5
    for lag in range(1, steps_num, 1):
        if lag <= 0:
            continue
        lag_name = str(lag).replace('-', 'minus_')  # todo minus, but it's not need
        lag_prev_name = str(lag-1).replace('-', 'minus_')

        df[f'u_in_lag_{lag_name}'] = df.groupby('breath_id')['u_in_lag_0'].shift(lag).fillna(0)
        df[f'u_in_diff_{lag_name}'] = (df[f'u_in_lag_{lag_prev_name}'] - df[f'u_in_lag_{lag_name}'])

        df[f'u_out_lag_{lag_name}'] = df.groupby('breath_id')['u_out_lag_0'].shift(lag).fillna(0)
        df[f'u_out_diff_{lag_name}'] = (df[f'u_out_lag_{lag_prev_name}'] - df[f'u_out_lag_{lag_name}'])

        df[f'time_step_lag_{lag_name}'] = df.groupby('breath_id')['time_step_lag_0'].shift(lag).fillna(0)
        df[f'time_diff_{lag_name}'] = (df[f'time_step_lag_{lag_prev_name}'] - df[f'time_step_lag_{lag_name}'])
        df[f'u_in_increase_{lag_name}'] = df[f'u_in_diff_{lag_name}'] * df[f'time_diff_{lag_name}']

    for step in range(1, steps_num, 1):  # maybe it's won't need to delete
        df.drop([f'u_in_lag_{step}'], axis=1, inplace=True)
        df.drop([f'time_step_lag_{step}'], axis=1, inplace=True)

    # last u_in
    df['last_value_u_in'] = df.groupby('breath_id')['u_in_lag_0'].transform('last')

    # first u_in, u_out
    df['u_in_first'] = df.groupby('breath_id')['u_in_lag_0'].first()
    df['u_out_first'] = df.groupby('breath_id')['u_out_lag_0'].first()

    # min max median mean
    for func in ['min', 'max', 'median', 'mean']:
        df[f'breath_id__u_in__{func}'] = df.groupby(['breath_id'])['u_in_lag_0'].transform(func)
        df[f'breath_id__u_out__{func}'] = df.groupby(['breath_id'])['u_out_lag_0'].transform(func)

    # from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
    df.loc[df['time_step_lag_0'] == 0, 'u_in_diff'] = 0
    df.loc[df['time_step_lag_0'] == 0, 'u_out_diff'] = 0

    # difference between the current value of u_in and the min, max, median values within the breath
    df['breath_id__u_in__diffmin'] = df.groupby(['breath_id'])['u_in_lag_0'].transform('min') - df['u_in_lag_0']
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in_lag_0'].transform('max') - df['u_in_lag_0']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in_lag_0'].transform('mean') - df['u_in_lag_0']

    # OHE
    # df['R_div_C'] = df['R'].div(df['C']) ухудшило
    df['R__C'] = df['R'].astype(str) + '__' + df['C'].astype(str)
    df = df.merge(pd.get_dummies(df['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
    df = df.merge(pd.get_dummies(df['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
    df = df.merge(pd.get_dummies(df['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'],
                                                                                                              axis=1)
    # CUMSUM https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/273974
    df['u_in_cumsum'] = df.groupby(['breath_id'])['u_in_lag_0'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step_lag_0'].cumsum()

    # очень маленькая прибавка, возможно и не надо?
    # time since last step
    df['time_step_diff'] = df.groupby('breath_id')['time_step_lag_0'].diff().fillna(0)
    # rolling window ts feats
    df['ewm_u_in_mean'] = df.groupby('breath_id')['u_in_lag_0'].ewm(halflife=9).mean().reset_index(level=0, drop=True)
    df['ewm_u_in_std'] = df.groupby('breath_id')['u_in_lag_0'].ewm(halflife=10).std().reset_index(level=0, drop=True)
    df['ewm_u_in_corr'] = df.groupby('breath_id')['u_in_lag_0'].ewm(halflife=15).corr().reset_index(level=0, drop=True)
    # rolling window of 15 periods
    df[['15_in_sum', '15_in_min', '15_in_max', '15_in_mean', '15_out_std']] = df.groupby('breath_id')[
        'u_in_lag_0'].rolling(window=15, min_periods=1).agg(
        {'15_in_sum': 'sum', '15_in_min': 'min', '15_in_max': 'max', '15_in_mean': 'mean',
         '15_in_std': 'std'}).reset_index(level=0, drop=True)
    df[['45_in_sum', '45_in_min', '45_in_max', '45_in_mean', '45_out_std']] = df.groupby('breath_id')[
        'u_in_lag_0'].rolling(window=45, min_periods=1).agg(
        {'45_in_sum': 'sum', '45_in_min': 'min', '45_in_max': 'max', '45_in_mean': 'mean',
         '45_in_std': 'std'}).reset_index(level=0, drop=True)

    df[['15_out_mean']] = df.groupby('breath_id')['u_out_lag_0'].rolling(window=15, min_periods=1).agg(
        {'15_out_mean': 'mean'}).reset_index(level=0, drop=True)

    # feature by u in or out (ideally - make 2 sep columns for each state) # dan
    df['u_in_partition_out_sum'] = df.groupby(['breath_id', 'u_out_lag_0'])['u_in_lag_0'].transform('sum')

    print(f'{"-" * 10}Num of features: {df.shape}{"-" * 10}')
    return df
