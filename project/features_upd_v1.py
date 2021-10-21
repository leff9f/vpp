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
    for lag in range(1, 5, 1):
        if lag == 0:
            continue
        lag_name = str(lag).replace('-', 'minus_')
        df[f'u_in_lag_{lag_name}'] = df.groupby('breath_id')['u_in'].shift(lag).fillna(0)
        df[f'u_out_lag_{lag_name}'] = df.groupby('breath_id')['u_out'].shift(lag).fillna(0)
        df[f'time_step_lag_{lag_name}'] = df.groupby('breath_id')['time_step'].shift(lag).fillna(0)
        df[f'time_diff_{lag_name}'] = (df['time_step'] - df[f'time_step_lag_{lag_name}'])
        df.drop([f'time_step_lag_{lag_name}'], axis=1, inplace=True)

    return df
