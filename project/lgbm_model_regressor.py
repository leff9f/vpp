import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from utilities import write_dict_csv
import matplotlib.pyplot as plt
import numpy as np

PARAMS = {
    'boosting_type': 'gbdt',
    'num_leaves': 1024,  # best 1024

    'objective': 'regression',
    'learning_rate': 0.25,

    "metric": 'mae',
    'n_jobs': -1,
    'min_data_in_leaf': 80,
    'max_bin': 210,
    # 'feature_fraction': 0.5, #0.4,
    # 'lambda_l1': 36, 'lambda_l2': 80,
    # 'max_depth': 16,
}


def train_and_evaluate_lgb_regressor(train, test, seed):
    print('data prepare')
    features = [col for col in train.columns if col not in {'id', 'pressure', 'breath_id'}]
    X = train[features]
    y = train['pressure']
    print('Create a KFold object')
    # folds = GroupKFold(n_splits=5)
    folds = GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=seed)
    # Iterate through each fold
    test_predictions = np.zeros(test.shape[0])
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (trn_ind, val_ind) in enumerate(folds.split(train, y, groups=train['breath_id'])):
        print(f'Started fold: {fold_n}')
        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        model = lgb.LGBMRegressor(**PARAMS, n_estimators=2000)
        model.fit(x_train, y_train,
                  eval_set=[(x_train, y_train), (x_val, y_val)],
                  verbose=100, early_stopping_rounds=25)
        score = mean_absolute_error(y_val, model.predict(x_val))

        test_predictions += model.predict(test[features]) / 5
        scores.append(score)

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = features
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

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

    # Return test predictions
    return test_predictions
