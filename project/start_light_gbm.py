from collections import defaultdict
from random import seed, sample
from typing import List, OrderedDict, DefaultDict, Tuple

from tqdm import tqdm
from utilities import read_csv_as_dicts, write_dict_csv

from data_plotter import Plotter
from lgbm_model import train_and_evaluate_lgb
from lgbm_model_regressor import train_and_evaluate_lgb_regressor
from lstm_model import train_and_evaluate_lstm

import pandas as pd
import numpy as np

from features import generate_features


class Stat:
    seed(0)

    def __init__(self, src_train_path: str, src_test_path: str, is_test: bool = True):
        self.src_train_path = src_train_path
        self.src_test_path = src_test_path
        self.is_test = is_test
        self.breath_ids = set()
        self.src_train = self.load_data(self.src_train_path)
        self.src_test = self.load_data(self.src_test_path)

    def load_data(self, path: str):
        loaded_data = []
        for num, row in tqdm(enumerate(read_csv_as_dicts(path)), desc=f'data loading from {path}'):
            loaded_data.append({key: float(el) for key, el in row.items()})
            if self.is_test and num == 499999:
                break
        return loaded_data

    def prepare_data_for_plot(self, breath_ids_num: int = 6) \
            -> DefaultDict[str, DefaultDict[str, Tuple[List, List]]]:
        breath_ids = sample(self.breath_ids, breath_ids_num)
        breath_ids = [b_id for b_id in breath_ids]
        data_sample = [
            row for row in tqdm(read_csv_as_dicts(self.src_train_path))
            if int(row['breath_id']) in breath_ids]
        source = {
            'R_ts': ['time_step', 'R'],
            'C_ts': ['time_step', 'C'],
            'uin_ts': ['time_step', 'u_in'],
            'uo_ts': ['time_step', 'u_out'],
            'p_ts': ['time_step', 'pressure'],
        }
        res = defaultdict(lambda: defaultdict(lambda: ([], [])))
        for el in data_sample:
            for name, coord in source.items():
                res[el["breath_id"]][f'{el["breath_id"]}{name}'][0].append(float(el[coord[0]]))
                res[el["breath_id"]][f'{el["breath_id"]}{name}'][1].append(float(el[coord[1]]))
        return res


def dict_to_array(data: dict):
    names = [key for key in data[0].keys()]
    formats = ['f8' for _ in data[0].keys()]
    dtype = dict(names=names, formats=formats)
    features = np.array([tuple(x.values()) for x in data], dtype=dtype)
    return features


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
    SEED0 = 159
    SEED1 = 2021

    is_load_features = False

    plt = Plotter('VPP', {})

    if not is_load_features:
        stat = Stat('../input/train.csv', '../input/test.csv', is_test=True)
        f_train, f_test = features_prepare(stat.src_train, stat.src_test,
                                           is_load_features=False, is_write_new_features=True)
    else:
        f_train, f_test = features_prepare(None, None,
                                           is_load_features=True, is_write_new_features=False)
    # submission = train_and_evaluate_lgb_regressor(f_train, f_test, SEED0)
    submission = train_and_evaluate_lstm(f_train, f_test)
    write_dict_csv('submission.csv', [{'id': num, 'pressure': p} for num, p in enumerate(submission, start=1)],
                   fieldnames=['id', 'pressure'])

    # for_plot = stat.prepare_data_for_plot(100)
    # plt.plot(lines=for_plot)
    # plt.save_layout('result.html')
