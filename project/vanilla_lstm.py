from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Bidirectional, Input, TimeDistributed, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from features_upd_v2_upd import generate_features
import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import io
from utilities import plot_validation_pressure, write_dict_csv, write_text_file
import datetime


PARAMS = {
    'batch_size': 256,
    'epochs': 200,
    'shuffle': False,
    'features': 'upd_v2_upd'
}


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def data_prepare(train, test):
    # get y, before scaling
    y = train['pressure'].to_numpy().reshape(-1, 80)

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
    train = train.reshape(-1, 80, train.shape[-1])
    test = test.reshape(-1, 80, train.shape[-1])
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
    desc = 'Bidir_LSTM_L400xBL300xBL200xBL100xD50SELUxD1'
    model.summary()
    return model, desc


def create_model_bidirectional_lstm_upd(n_steps, n_features):
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Bidirectional(LSTM(800, return_sequences=True)),
        Bidirectional(LSTM(512, return_sequences=True)),
        Bidirectional(LSTM(256, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dense(128, activation='elu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    desc = 'Bidir_LSTM_BL800xBL512xBL256xBL128xD128ELUxD1'
    model.summary()
    return model, desc


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


def train_and_evaluate_v_lstm(train, test):
    train, test, y, features = data_prepare(train, test)

    kf = KFold(n_splits=5, shuffle=True, random_state=159)
    test_preds = []
    val_preds = []
    val_true = []
    scores = []
    date_time = datetime.datetime.now()
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, y)):
        print(f'{"-"*10}Fold number {fold+1}{"-"*10}')

        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = y[train_idx], y[test_idx]

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 200 * ((len(train) * 0.8) / 512), 1e-5)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1, restore_best_weights=True)

        model, model_desc = create_model_bidirectional_lstm_upd(n_steps=train.shape[1], n_features=train.shape[2])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(X_train, y_train,
                  validation_data=(X_valid, y_valid),
                  epochs=PARAMS['epochs'],
                  batch_size=PARAMS['batch_size'],
                  callbacks=[es, tf.keras.callbacks.LearningRateScheduler(scheduler), tensorboard_callback]
                  )

        # save model
        model.save(f'artefacts/{model_desc}_f_{fold}_{PARAMS["features"]}.h5')

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
            model_summary = get_model_summary(model).replace('Model', f'Model started: {date_time}')

        del X_train, X_valid, y_train, y_valid, model
        gc.collect()

    # plot validation pressure
    plot_validation_pressure(val_preds, val_true, f'artefacts/vanilla_validation_result_{PARAMS["features"]}.html')

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

    write_dict_csv('artefacts/vanilla_params_and_features_w_cv_score.csv', [params], list(params.keys()), mode='a')
    write_text_file('artefacts/vanilla_params_model.csv', model_summary, mode='a')

    # Mean submission
    submission = pd.read_csv('../input/sample_submission.csv')
    submission["pressure"] = sum(test_preds) / 5  # test_preds[1]
    submission.to_csv(f'artefacts/submission_mean_{PARAMS["features"]}.csv', index=False)

    # Median submission
    submission["pressure"] = np.median(np.vstack(test_preds), axis=0)

    # Round predictions (Post Preprocessing)
    train = pd.read_csv('../input/train.csv')
    pressure_unique = np.array(sorted(train['pressure'].unique()))
    submission['pressure'] = submission['pressure'].map(lambda x: pressure_unique[np.abs(pressure_unique - x).argmin()])
    submission.to_csv(f'artefacts/submission_post_preprocessing_{PARAMS["features"]}.csv', index=False)
