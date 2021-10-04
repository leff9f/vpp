import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from utilities import write_dict_csv
import matplotlib.pyplot as plt
import numpy as np

# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))


def scaling_data(X, y):
    X_feat = X[[x for x in X.columns if x != 'breath_id']]
    y_feat = y[['pressure']]

    # Fit the scaler using available training data
    input_scaler = scaler_x.fit(X_feat)
    output_scaler = scaler_y.fit(y_feat)

    # Apply the scaler to training data
    x_norm = input_scaler.transform(X_feat)
    y_norm = output_scaler.transform(y_feat)
    y_norm = np.append(y_norm, y['breath_id'].to_numpy()[:, np.newaxis], axis=1)

    return x_norm, y_norm


def scaling_test_data(X):
    X_feat = X[[x for x in X.columns if x != 'breath_id']]

    # Fit the scaler using available training data
    input_scaler = scaler_x.fit(X_feat)

    # Apply the scaler to training data
    x_norm = input_scaler.transform(X_feat)
    x_norm = np.append(x_norm, X['breath_id'].to_numpy()[:, np.newaxis], axis=1)
    return x_norm


def create_dataset(x, y):
    xs, ys = [], []
    breath_ids = Counter(y[:, 1])

    start = 0
    for b_id, interval_num in breath_ids.items():
        xs.append(x[start: start+interval_num, :])
        ys.append(y[start: start+interval_num, 0])
        start += interval_num

    return np.array(xs), np.array(ys)


def create_test_dataset(x):
    xs = []
    breath_ids = Counter(x[:, -1])

    start = 0
    for b_id, interval_num in breath_ids.items():
        xs.append(x[start: start+interval_num, :-1])
        start += interval_num

    return np.array(xs)


# Create BiLSTM model
def create_model_bilstm(X_train):
    model = Sequential()
    # First layer of BiLSTM
    model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    # Second layer of BiLSTM
    model.add(Bidirectional(LSTM(units=64)))
    model.add(Dense(80))
    # Compile model
    model.compile(loss='mae', optimizer='adam')
    return model


def prepare_dataset(X, y, trn_ind, val_ind, test):
    x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
    x_train, y_train = scaling_data(x_train, y_train)
    x_val, y_val = scaling_data(x_val, y_val)
    test = scaling_test_data(test)

    x_train_ds, y_train_ds = create_dataset(x_train, y_train)
    x_val_ds, y_val_ds = create_dataset(x_val, y_val)
    test_ds = create_test_dataset(test)
    print(f'x_train.shape: {x_train_ds.shape} '
          f'y_train.shape: {y_train_ds.shape} '
          f'x_val.shape: {x_val_ds.shape} '
          f'y_val.shape: {y_val_ds.shape} '
          f'test.shape: {test_ds.shape}')
    return x_train_ds, y_train_ds, x_val_ds, y_val_ds, test_ds


# Fit BiLSTM, LSTM and GRU
def fit_model(model, X_train, y_train, X_test, y_test):
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, verbose=0,
        mode='min', restore_best_weights=True)

    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=7, verbose=0,
        mode='min')

    history = model.fit(X_train,
                        y_train,
                        batch_size=800,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        validation_batch_size=len(y_test),
                        shuffle=True,
                        callbacks=[es, plateau])
    return history


def prediction(model, X_test):
    predict = model.predict(X_test)
    predict = scaler_y.inverse_transform(predict).flatten()
    return predict


def train_and_evaluate_lstm(train, test):
    print('data prepare')
    features = [col for col in train.columns if col not in {'id', 'pressure'}]
    test_featured = [col for col in test.columns if col not in {'id'}]
    X = train[features]
    y = train[['pressure', 'breath_id']]
    print('Create a KFold object')
    folds = GroupKFold(n_splits=5)
    # Iterate through each fold
    test_predictions = np.zeros(test.shape[0])

    # self check validation
    scores = []
    for fold_n, (trn_ind, val_ind) in enumerate(folds.split(train, y, groups=train['breath_id'])):
        x_train_ds, y_train_ds, x_val_ds, y_val_ds, test_ds = prepare_dataset(X, y, trn_ind, val_ind,
                                                                              test[test_featured])

        model_bilstm = create_model_bilstm(x_train_ds)
        history_bilstm = fit_model(model_bilstm, x_train_ds, y_train_ds, x_val_ds, y_val_ds)

        score = mean_absolute_error(y_val_ds, model_bilstm.predict(x_val_ds))
        scores.append(score)

        # Note that I have to use scaler_y
        y_val = scaler_y.inverse_transform(y_val_ds)
        y_train = scaler_y.inverse_transform(y_train_ds)

        test_predictions += prediction(model_bilstm, test_ds) / 5

    cv_mean_score = np.mean(scores)
    cv_std_score = np.std(scores)
    print(f'CV mean score: {cv_mean_score}, std: {cv_std_score}.')

    # TODO Plot
    print('Plotting feature importances...')

    # write config and CV score
    params = {}
    params.update({
        'CV mean score': cv_mean_score,
        'CV std score': cv_std_score,
        'num rows': train.shape[0],
        'features': ', '.join(features)
    })

    write_dict_csv('params_and_features_lstm_cv_score.csv', [params], list(params.keys()), mode='a')

    # Return test predictions
    return test_predictions
