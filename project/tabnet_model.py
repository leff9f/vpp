from pytorch_tabnet.tab_model import TabNetRegressor
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Bidirectional, Input, TimeDistributed, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from features_upd_v2 import generate_features
import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import io
from utilities import plot_validation_pressure, write_dict_csv, write_text_file
import datetime

PARAMS = {'batch_size': 256,
          'epochs': 200,
          'shuffle': False,
          }


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def data_prepare(train, test):
    # get y, before scaling
    y = train['pressure'].to_numpy().reshape(-1, 1)

    # generate features
    train = generate_features(train)
    test = generate_features(test)

    # # drop unused features
    train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
    test.drop(['id', 'breath_id'], axis=1, inplace=True)
    # train.drop(['pressure', 'id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
    #             'breath_id_lag2same', 'u_out_lag2'], axis=1, inplace=True)
    # test.drop(['id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
    #                   'breath_id_lag2same', 'u_out_lag2'], axis=1)

    # features
    features = train.columns

    # Scale features
    rb = RobustScaler()
    rb.fit(train)
    train = rb.transform(train)
    test = rb.transform(test)
    # train = train.reshape(-1, 80, train.shape[-1])
    # test = test.reshape(-1, 80, train.shape[-1])
    gc.collect()
    return train, test, y, features


def create_model_vlstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='selu', input_shape=(n_steps, n_features), return_sequences=True))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()
    return model


def create_model_lstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(400, input_shape=(n_steps, n_features), return_sequences=True))
    model.add(LSTM(300, return_sequences=True))
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dense(50, activation='selu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()
    return model


def create_model_bidirectional_lstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(400, input_shape=(n_steps, n_features), return_sequences=True))
    model.add(Bidirectional(LSTM(300, return_sequences=True)))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dense(50, activation='selu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()
    return model


def create_model_bidirectional_lstm_upd(n_steps, n_features):
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Bidirectional(LSTM(700, return_sequences=True)),
        Bidirectional(LSTM(512, return_sequences=True)),
        Bidirectional(LSTM(256, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dense(128, activation='elu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()
    return model


def create_model_cnn_lstm(n_steps, n_features):
    # need to reformat data, for all breath make sample with shift
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='selu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()
    return model


def train_and_evaluate_tabnet(train, test):
    train, test, y, features = data_prepare(train, test)

    kf = KFold(n_splits=6, shuffle=True, random_state=159)
    test_preds = []
    val_preds = []
    val_true = []
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(train, y)):
        print(f'{"-" * 10}Fold number {fold + 1}{"-" * 10}')

        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = y[train_idx], y[test_idx]

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 200 * ((len(train) * 0.8) / 512), 1e-5)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1, restore_best_weights=True)

        model = TabNetRegressor()
        model.fit(
            X_train, y_train,
            max_epochs=200,
            patience=20,
            batch_size=72000,
            virtual_batch_size=240,
            num_workers=0,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['mae']
        )
        test_preds.append(model.predict(test).squeeze().reshape(-1, 1).squeeze())

        # get validation pressure sample for result
        pred_valid = model.predict(X_valid).squeeze()
        random = np.random.choice(pred_valid.shape[0], 10, replace=False)
        val_preds.extend(pred_valid[random])
        val_true.extend(y_valid[random])

        # get result validation scores
        score = mean_absolute_error(y_valid, pred_valid)
        scores.append(score)

        # get model summary
        if fold == 0:
            date_time = datetime.datetime.now()

        del X_train, X_valid, y_train, y_valid, model
        gc.collect()

        # plot validation pressure
    plot_validation_pressure(val_preds, val_true, 'vanilla_validation_result.html')

    cv_mean_score = np.mean(scores)
    cv_std_score = np.std(scores)
    print(f'CV mean score: {cv_mean_score}, std: {cv_std_score}.')

    # write config and CV score
    params = PARAMS
    params.update({
        'CV mean score': cv_mean_score,
        'CV std score': cv_std_score,
        'num breaths': train.shape[0],
        'num features': train.shape[1],
        'features': ', '.join(features),
        'date time': date_time
    })

    write_dict_csv('vanilla_params_and_features_w_cv_score.csv', [params], list(params.keys()), mode='a')
    # write_text_file('vanilla_params_model.csv', model_summary, mode='a')

    # Mean submission
    submission = pd.read_csv('../input/sample_submission.csv')
    submission["pressure"] = sum(test_preds) / 5  # test_preds[1]
    submission.to_csv('submission_mean_v2.csv', index=False)

    # Median submission
    submission["pressure"] = np.median(np.vstack(test_preds), axis=0)

    # Round predictions (Post Preprocessing)
    train = pd.read_csv('../input/train.csv')
    pressure_unique = np.array(sorted(train['pressure'].unique()))
    submission['pressure'] = submission['pressure'].map(lambda x: pressure_unique[np.abs(pressure_unique - x).argmin()])
    submission.to_csv('submission_post_preprocessing_v2.csv', index=False)