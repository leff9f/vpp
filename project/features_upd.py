import pandas as pd
import numpy as np


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['cross'] = df['u_in'] * df['u_out']
    df['cross2'] = df['time_step'] * df['u_out']

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] = df['u_in_cumsum'] / df['count']

    df['breath_id_lag'] = df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2'] = df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame'] = np.select([df['breath_id_lag'] == df['breath_id']], [1], 0)
    df['breath_id_lag2same'] = np.select([df['breath_id_lag2'] == df['breath_id']], [1], 0)

    # create lag in data - большее число сдвигов
    for lag in range(-3, 3, 1):
        if lag == 0:
            continue
        df[f'u_in_lag_{str(lag)}'] = df.groupby('breath_id')['u_in'].shift(lag).fillna(0)
        df[f'u_out_lag_{str(lag)}'] = df.groupby('breath_id')['u_out'].shift(lag).fillna(0)

    # min max median mean ~0.2
    for func in ['min', 'max', 'median', 'mean']:
        df[f'breath_id__u_in__{func}'] = df.groupby(['breath_id'])['u_in'].transform(func)
        df[f'breath_id__u_out__{func}'] = df.groupby(['breath_id'])['u_out'].transform(func)

    # last u_in ~0.02
    df['last_value_u_in'] = df.groupby('breath_id')['u_in'].transform('last')
    # first u_in, u_out
    df['u_in_first'] = df.groupby('breath_id')['u_in'].first()
    df['u_out_first'] = df.groupby('breath_id')['u_out'].first()

    # difference between consecutive (последовательными) values ~0.08
    for lag in ['1', '2']:
        df[f'u_in_diff_{lag}'] = df['u_in'] - df[f'u_in_lag_{lag}']
        df[f'u_out_diff_{lag}'] = df['u_out'] - df[f'u_out_lag_{lag}']

    # difference between the current value of u_in and the min, max, median values within the breath ~0.03
    df['breath_id__u_in__diffmin'] = df[f'breath_id__u_in__min'] - df['u_in']
    df['breath_id__u_in__diffmax'] = df[f'breath_id__u_in__max'] - df['u_in']
    df['breath_id__u_in__diffmean'] = df[f'breath_id__u_in__mean'] - df['u_in']
    df['breath_id__u_in__diffmedian'] = df[f'breath_id__u_in__median'] - df['u_in']

    df = df.fillna(0)  # дополнить к данным со смещениями 0 значения
    print(f'{"*"*20}Features num: {df.shape}{"*"*20}')
    return df
