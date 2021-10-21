import pandas as pd
import math
import numpy as np


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

    steps_num = 5
    for lag in range(1, steps_num, 1):
        if lag <= 0:
            continue
        lag_name = str(lag).replace('-', 'minus_')  # todo minus, but it's not need
        lag_prev_name = str(lag-1).replace('-', 'minus_')

        df[f'u_in_lag_{lag_name}'] = df.groupby('breath_id')['u_in_lag_0'].shift(lag).fillna(0)
        df[f'u_in_diff_{lag_name}'] = (df[f'u_in_lag_{lag_prev_name}'] - df[f'u_in_lag_{lag_name}'])
        df[f'u_in_full_diff_{lag_name}'] = (df[f'u_in_lag_0'] - df[f'u_in_lag_{lag_name}'])

        df[f'time_step_lag_{lag_name}'] = df.groupby('breath_id')['time_step_lag_0'].shift(lag).fillna(0)
        df[f'time_diff_{lag_name}'] = (df[f'time_step_lag_{lag_prev_name}'] - df[f'time_step_lag_{lag_name}'])
        df[f'time_full_diff_{lag_name}'] = (df[f'u_in_lag_0'] - df[f'time_step_lag_{lag_name}'])

        df[f'u_in_increase_{lag_name}'] = df[f'u_in_diff_{lag_name}'] * df[f'time_diff_{lag_name}']
        df[f'u_in_diff_v_{lag_name}'] = df[f'u_in_diff_{lag_name}']*np.ones(df[f'time_diff_{lag_name}'].shape) - \
            np.ones(df[f'time_diff_{lag_name}'].shape)/np.exp(df[f'time_diff_{lag_name}']/(df['R']*df['C']))

    # # maybe it's won't need to delete
    # for step in range(1, steps_num, 1):
    #     df.drop([f'u_in_lag_{step}'], axis=1, inplace=True)
    #     df.drop([f'time_step_lag_{step}'], axis=1, inplace=True)

    df[f'u_in_max'] = df.groupby('breath_id')['u_in_lag_0'].transform('max')
    df[f'u_in_median'] = df.groupby('breath_id')['u_in_lag_0'].transform('median')

    print(f'{"-" * 10}Num of features: {df.shape}{"-" * 10}')
    return df
