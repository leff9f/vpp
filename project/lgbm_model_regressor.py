import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from utilities import write_dict_csv, feval_mae
from sklearn.preprocessing import RobustScaler
from features_upd_v3 import generate_features
import matplotlib.pyplot as plt
import numpy as np
import gc

PARAMS = {
    'boosting_type': 'gbdt',
    'num_leaves': 1024,  # best 1024

    'objective': 'regression',
    'learning_rate': 0.25,  # best 0.3

    "metric": 'mae',
    'n_jobs': -1,
    'min_data_in_leaf': 160,  # best 32
    'max_bin': 210, #196,
    # 'feature_fraction': 0.5, #0.4,
    # 'lambda_l1': 36, 'lambda_l2': 80,
    # 'max_depth': 16,
}


def data_prepare(train, test):
    # get y, before scaling
    y = train['pressure'].to_numpy()

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
    gc.collect()
    return train, test, y, features


def train_and_evaluate_lgb_regressor(train, test):
    train, test, y, features = data_prepare(train, test)
    print('Create a KFold object')
    kf = KFold(n_splits=5, shuffle=True, random_state=159)
    # Iterate through each fold
    test_preds = []
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_idx, test_idx) in enumerate(kf.split(train, y)):
        print(f'{"*"*20}Training fold {fold_n + 1}{"*"*20}')
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(**PARAMS, n_estimators=2300)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  verbose=100,
                  early_stopping_rounds=25)
        score = mean_absolute_error(y_valid, model.predict(X_valid))
        scores.append(score)

        test_preds.append(model.predict(test).squeeze().reshape(-1, 1).squeeze())

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = features
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        del X_train, X_valid, y_train, y_valid
        gc.collect()

        print(f'{"*" * 20}mae score: {np.mean(score)}{"*" * 20}')

    cv_mean_score = np.mean(scores)
    cv_std_score = np.std(scores)
    print(f'CV mean score: {cv_mean_score}, std: {cv_std_score}.')

    print('Plotting feature importances...')
    ax = lgb.plot_importance(model, max_num_features=20)
    plt.savefig('feature_importance.png')

    # write config and CV score
    params = PARAMS
    params.update({
        'CV mean score': cv_mean_score,
        'CV std score': cv_std_score,
        'num rows': train.shape[0],
        'features': ', '.join(features)
    })

    write_dict_csv('params_and_features_w_cv_score.csv', [params], list(params.keys()), mode='a')

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
