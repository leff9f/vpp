from collections import defaultdict
from random import seed, sample
from typing import List, DefaultDict, Tuple

from tqdm import tqdm
import pandas as pd
from utilities import read_csv_as_dicts

from data_plotter import Plotter
from features_upd import generate_features


class Stat:
    seed(0)

    def __init__(self, src_train_path: str, src_test_path: str, nrows: int = -1):
        self.src_train_path = src_train_path
        self.src_test_path = src_test_path
        self.nrows = nrows
        self.breath_ids = set([row['breath_id']
                               for row in tqdm(read_csv_as_dicts(src_train_path, nrows=self.nrows),
                                               desc='get breath ids')])

    def load_data(self, path: str):
        loaded_data = []
        for num, row in tqdm(enumerate(read_csv_as_dicts(path))):
            loaded_data.append({key: float(el) for key, el in row.items()})
        return loaded_data

    def prepare_data_for_plot(self, breath_ids_num: int = 6) \
            -> DefaultDict[str, DefaultDict[str, Tuple[List, List]]]:
        breath_ids = sample(self.breath_ids, breath_ids_num)
        breath_ids = [b_id for b_id in breath_ids]
        data_sample = [
            row for row in tqdm(read_csv_as_dicts(self.src_train_path, nrows=self.nrows))
            if row['breath_id'] in breath_ids]

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

    def prepare_features_for_plot(self, breath_ids_num: int = 6) \
            -> DefaultDict[str, DefaultDict[str, Tuple[List, List]]]:
        data_for_features = pd.read_csv(self.src_train_path, nrows=self.nrows)
        data_w_features = generate_features(data_for_features)
        features = {
            'R_ts': ['time_step', 'R'],
            'C_ts': ['time_step', 'C'],
            'uin_ts': ['time_step', 'u_in'],
            'uo_ts': ['time_step', 'u_out'],
            'p_ts': ['time_step', 'pressure']
            # 'area': ['time_step', 'area'],
            # 'cross': ['time_step', 'cross'],
            # 'cross2': ['time_step', 'cross2'],
            # 'u_in_cumsum': ['time_step', 'u_in_cumsum'],
            # 'count': ['time_step', 'count'],
            # 'breath_id_lag': ['time_step', 'breath_id_lag'],
            # 'breath_id_lagsame': ['time_step', 'breath_id_lagsame'],
            # 'u_in_lag': ['time_step', 'u_in_lag'],
            # 'u_in': ['time_step', 'u_in'],
        }
        b_ids = data_w_features['breath_id'].unique()
        b_ids = sample(list(b_ids), breath_ids_num)
        res = defaultdict(lambda: defaultdict(lambda: ([], [])))
        for b_id in b_ids:
            for name, coord in features.items():
                res[str(b_id)][f'{str(b_id)}_{name}'][0].extend(
                    data_w_features.loc[data_w_features['breath_id'] == b_id][coord[0]].tolist()
                )
                res[str(b_id)][f'{str(b_id)}_{name}'][1].extend(
                    data_w_features.loc[data_w_features['breath_id'] == b_id][coord[1]].tolist()
                )
        return res


if __name__ == '__main__':
    plt = Plotter('VPP', {})
    stat = Stat('../input/train.csv', '../input/test.csv', nrows=9999)
    for_plot = stat.prepare_features_for_plot(10)
    print('Data prepared, plotting')
    plt.plot(lines=for_plot)
    plt.save_layout('result.html')
    print('Job done!')
