from typing import List
import os
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
from data_plotter import Plotter


def read_csv_as_dicts(path, nrows=-1, delimiter=','):
    with open(path) as f:
        lines = csv.DictReader(f, delimiter=delimiter)
        for num, row in enumerate(lines):
            if num == nrows:
                break
            yield row


def write_dict_csv(path, data, fieldnames: List[str], mode='w'):
    header = True
    if mode == 'a' and os.path.isfile(path):
        header = False

    with open(path, mode=mode) as f:
        writer = csv.DictWriter(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        if header:
            writer.writeheader()
        for el in data:
            writer.writerow(el)


def write_text_file(path, data, mode='w'):
    with open(os.path.join(path), mode) as f:
        f.write(data)


# Mean average error
def mae(y_true, y_pred):
    return np.sum(np.absolute(y_pred-y_true))/y_pred.size


# Function to early stop with root mean squared percentage error
def feval_mae(y_pred, lgb_train):
    y_true = lgb_train.get_label().to_numpy()
    return 'MAE', mean_absolute_error(y_true, y_pred), False


def plot_validation_pressure(val_preds, val_true, file_name):
    to_plot = defaultdict(lambda: defaultdict(lambda: ([], [])))
    steps = [x for x in range(80)]
    for num, (v_pred, v_true) in enumerate(zip(val_preds, val_true)):
        to_plot[str(num)]['val_pred'][0].extend(steps)
        to_plot[str(num)]['val_pred'][1].extend(v_pred.tolist())
        to_plot[str(num)]['val_true'][0].extend(steps)
        to_plot[str(num)]['val_true'][1].extend(v_true.tolist())

    plt = Plotter('VPP', {})
    plt.plot(lines=to_plot)
    plt.save_layout(file_name)