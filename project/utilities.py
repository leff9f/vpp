from typing import List
import os
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error


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


# Mean average error
def mae(y_true, y_pred):
    return np.sum(np.absolute(y_pred-y_true))/y_pred.size


# Function to early stop with root mean squared percentage error
def feval_mae(y_pred, lgb_train):
    y_true = lgb_train.get_label().to_numpy()
    return 'MAE', mean_absolute_error(y_true, y_pred), False