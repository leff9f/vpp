from vanilla_lstm import train_and_evaluate_v_lstm
import pandas as pd
import numpy as np

from features import generate_features


def dict_to_array(data: dict):
    names = [key for key in data[0].keys()]
    formats = ['f8' for _ in data[0].keys()]
    dtype = dict(names=names, formats=formats)
    features = np.array([tuple(x.values()) for x in data], dtype=dtype)
    return features


# is deprecated
def features_prepare(train, test, is_load_features: bool = False, is_write_new_features: bool = False):
    if is_load_features:
        train = pd.read_pickle('../input/train_prepared_features.pkl', compression='infer')
        test = pd.read_pickle('../input/test_prepared_features.pkl', compression='infer')
        print('features loaded')
        return train, test

    train = generate_features(pd.DataFrame(train))
    test = generate_features(pd.DataFrame(test))
    print('features created')

    # # исключить фазу выдоха
    # train = train.loc[train["u_out"] != 1]

    if is_write_new_features:
        pd.to_pickle(train, '../input/train_prepared_features.pkl')
        pd.to_pickle(test, '../input/test_prepared_features.pkl')
        print('features writen')

    return train, test


if __name__ == '__main__':
    src_train_path = '../input/train.csv'
    src_test_path = '../input/test.csv'
    nrows = None  # if need some small test set, set int number, for all data use None
    src_train = pd.read_csv(src_train_path, nrows=nrows)
    src_test = pd.read_csv(src_test_path, nrows=nrows)
    SEED0 = 159
    train_and_evaluate_v_lstm(src_train, src_test)
