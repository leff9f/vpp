from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from collections import Counter, defaultdict
import gc
import itertools
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Softmax
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from utilities import write_dict_csv
import matplotlib.pyplot as plt
import numpy as np

# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))


params = {'batch_size': 256,
          'epochs': 5,
          'validation_batch_size': 256,
          'shuffle': False,
          }


def plot_loss(history, model_name, fold_n):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.savefig(f'{model_name}_loss_{fold_n}.png')


def plot_future(prediction, model_name, y_test):
    plt.figure(figsize=(10, 6))

    range_future = len(prediction)

    plt.plot(np.arange(range_future), np.array(y_test), label='True Future')
    plt.plot(np.arange(range_future), np.array(prediction), label='Prediction')

    plt.title('True future vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Breath id')
    plt.ylabel('Pressure')
    plt.savefig(f'{model_name}_predict.png')


def scaling_data(X, y):
    X_feat = X[[x for x in X.columns if x != 'breath_id']]
    y_feat = y[['pressure']]

    # Fit the scaler using available training data
    # Apply the scaler to training data
    input_scaler = scaler_x.fit(X_feat)
    x_norm = np.float32(input_scaler.transform(X_feat))

    del input_scaler, X_feat, y_feat
    gc.collect()

    y_norm = np.float32(np.append(y, y['breath_id'].to_numpy()[:, np.newaxis], axis=1))

    return x_norm, y_norm


def scaling_test_data(X):
    X_feat = X[[x for x in X.columns if x != 'breath_id']]

    # Fit the scaler using available training data
    input_scaler = scaler_x.fit(X_feat)

    # Apply the scaler to training data
    x_norm = np.float32(input_scaler.transform(X_feat))

    del input_scaler, X_feat
    gc.collect()

    x_norm = np.float32(np.append(x_norm, X['breath_id'].to_numpy()[:, np.newaxis], axis=1))
    return x_norm


def create_dataset(x, y):
    xs, ys = [], []
    breath_ids = Counter(y[:, 1])

    start = 0
    for b_id, interval_num in breath_ids.items():
        xs.append(x[start: start + interval_num, :])
        ys.append(y[start: start + interval_num, 0])
        start += interval_num

    return np.float32(np.array(xs)), np.float32(np.array(ys))


def create_test_dataset(x):
    xs = []
    breath_ids = Counter(x[:, -1])

    start = 0
    for b_id, interval_num in breath_ids.items():
        xs.append(x[start: start + interval_num, :-1])
        start += interval_num

    return np.float32(np.array(xs))


def prepare_dataset(X, y, trn_ind, val_ind):
    x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
    x_train, y_train = scaling_data(x_train, y_train)
    x_val, y_val = scaling_data(x_val, y_val)

    # for group-fold
    x_train_ds, y_train_ds = create_dataset(x_train, y_train)
    del x_train, y_train
    gc.collect()
    x_val_ds, y_val_ds = create_dataset(x_val, y_val)
    del x_val, y_val
    gc.collect()

    # for k-fold
    # x_train_ds, y_train_ds = x_train.reshape(-1, 80, x_train.shape[-1]), y_train[:, -1].reshape(-1, 80)
    # x_val_ds, y_val_ds = x_val.reshape(-1, 80, x_val.shape[-1]), y_val[:, -1].reshape(-1, 80)
    print(f'x_train.shape: {x_train_ds.shape} '
          f'y_train.shape: {y_train_ds.shape} '
          f'x_val.shape: {x_val_ds.shape} '
          f'y_val.shape: {y_val_ds.shape}')
    return x_train_ds, y_train_ds, x_val_ds, y_val_ds


def prepare_test_dataset(test):
    test = scaling_test_data(test)
    test_ds = create_test_dataset(test)
    print(f'test.shape: {test_ds.shape}')
    return test_ds


def prediction(model, X_test):
    predict = model.predict(X_test)
    predict = scaler_y.inverse_transform(predict[:, :, -1]).flatten()
    return predict


def split_weighted_folds(X, n_folds, group_len, classes):
    # unique values in classes(columns)
    unique_cls_vals = []
    for cls in classes:
        unique_cls_vals.append(X[cls].unique())

    # create unique values pairs
    def create_all_pairs(*seqs):
        if not seqs:
            return [[]]
        else:
            return [[x] + p for x in seqs[0] for p in create_all_pairs(*seqs[1:])]

    unique_classes = create_all_pairs(unique_cls_vals[0], unique_cls_vals[1])

    # split by unique indexes
    unique_indexes = []
    for cls_name, values in zip(itertools.cycle([classes]), unique_classes):
        indexes = X.index[(X[cls_name[0]] == values[0]) & (X[cls_name[1]] == values[1])].tolist()
        unique_indexes.append(indexes)

    random.seed(159)
    folds = defaultdict(list)
    for u_ind in unique_indexes:
        indexes_by_folds = [u_ind[i*group_len:(i+1)*group_len] for i in range(int(len(u_ind)/group_len))]
        random.shuffle(indexes_by_folds)
        indexes_by_folds = [indexes_by_folds[i::n_folds] for i in range(n_folds)]
        for fold in range(n_folds):
            folds[fold].extend([sub_ind for sub_list in indexes_by_folds[fold] for sub_ind in sub_list])
    for fold in folds.values():
        yield list(set(X.index.tolist()).difference(fold)), fold


# Create BiLSTM model
def create_model_bilstm(X_train):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        # keras.layers.Bidirectional(keras.layers.LSTM(400, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(300, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(250, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(150, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
        #             keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.Dense(50, activation='selu'),
        #             keras.layers.Dropout(0.1),
        keras.layers.Dense(1),
    ])
    model.summary()
    model.compile(optimizer="adam", loss="mean_absolute_error")

    return model


# Fit BiLSTM, LSTM and GRU
def fit_model(model, X_train, y_train, X_test, y_test):
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, verbose=0,
        mode='min', restore_best_weights=True)

    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=7, verbose=0,
        mode='min')

    scheduler = ExponentialDecay(1e-3, 400 * ((len(X_train) * 0.8) / params['batch_size']), 1e-5)
    lr = LearningRateScheduler(scheduler, verbose=1)

    history = model.fit(X_train,
                        y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=(X_test, y_test),
                        validation_batch_size=params['validation_batch_size'],
                        shuffle=params['shuffle'],
                        callbacks=[lr])
    return history


def train_and_evaluate_lstm(train, test):
    gpu_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    print(gpu_available)

    print('data prepare')
    features = [col for col in train.columns if col not in {'id', 'pressure'}]
    test_featured = [col for col in test.columns if col not in {'id'}]
    X = train[features]
    y = train[['pressure', 'breath_id']]
    print('Create a KFold object')

    # get pressures
    all_pressure = np.sort(train.pressure.unique())
    PRESSURE_MIN = all_pressure[0].item()
    PRESSURE_MAX = all_pressure[-1].item()
    PRESSURE_STEP = (all_pressure[1] - all_pressure[0]).item()

    folds = GroupKFold(n_splits=5)
    test_predictions = []
    submission = np.zeros(test.shape[0])

    # self check validation
    scores = []
    test_ds = prepare_test_dataset(test[test_featured])

    del test
    del train
    gc.collect()

    # for fold_n, (trn_ind, val_ind) in enumerate(folds.split(X, y, groups=X['breath_id'])):
    for fold_n, (trn_ind, val_ind) in enumerate(split_weighted_folds(X, 5, 80, ['R', 'C'])):
        x_train_ds, y_train_ds, x_val_ds, y_val_ds = prepare_dataset(X, y, trn_ind, val_ind)

        model_bilstm = create_model_bilstm(x_train_ds)
        history_bilstm = fit_model(model_bilstm, x_train_ds, y_train_ds, x_val_ds, y_val_ds)
        plot_loss(history_bilstm, 'BiLSTM', fold_n)

        score = mean_absolute_error(y_val_ds, model_bilstm.predict(x_val_ds)[:, :, -1])
        scores.append(score)

        print(f'CV score: {score}')

        test_predictions.append(model_bilstm.predict(test_ds)[:, :, -1].flatten())

    # plot_future(test_predictions, 'bilstm', y_val)
    cv_mean_score = np.mean(scores)
    cv_std_score = np.std(scores)
    print(f'CV mean score: {cv_mean_score}, std: {cv_std_score}.')

    # write config and CV score
    params.update({
        'CV mean score': cv_mean_score,
        'CV std score': cv_std_score,
        'num rows': len(features),
        'features': ', '.join(features)
    })

    write_dict_csv('params_and_features_lstm_cv_score.csv', [params], list(params.keys()), mode='a')

    # median round predict
    submission = np.median(np.vstack(test_predictions), axis=0)
    submission = np.round((submission - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
    submission = np.clip(submission, PRESSURE_MIN, PRESSURE_MAX)

    # Return submission
    return submission
