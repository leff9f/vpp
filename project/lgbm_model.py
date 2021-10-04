import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error
from utilities import feval_mae
import matplotlib.pyplot as plt

SEED = 0
PARAMS = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'num_leaves': 1024,
    'max_depth': -1,
    'max_bin': 100,
    'min_data_in_leaf': 32,
    'learning_rate': 0.3,
    # 'subsample': 0.72,
    # 'subsample_freq': 4,
    # 'feature_fraction': 0.5,
    # 'lambda_l1': 0.5,
    # 'lambda_l2': 1.0,
    # 'categorical_column': [0],
    # 'seed': SEED,
    # 'feature_fraction_seed': SEED,
    # 'bagging_seed': SEED,
    # 'drop_seed': SEED,
    # 'data_random_seed': SEED,
    'n_jobs': -1
    # 'verbose': -1
}


# def train_and_evaluate_lgb(train, test, params):
#     print('data prepare')
#     features = [col for col in train.columns if col not in {'id', 'pressure', 'breath_id'}]
#     y = train['pressure']
#     oof_predictions = np.zeros(train.shape[0])
#     # Create test array to store predictions
#     test_predictions = np.zeros(test.shape[0])
#     print('Create a KFold object')
#     kfold = KFold(n_splits=5, random_state=SEED, shuffle=True)
#     # Iterate through each fold
#     for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
#         print(f'Training fold {fold + 1}')
#         x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
#         y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
#         # Root mean squared percentage error weights
#         train_weights = 1 / np.square(y_train)
#         val_weights = 1 / np.square(y_val)
#         train_dataset = lgb.Dataset(x_train[features], y_train, weight=train_weights)
#         val_dataset = lgb.Dataset(x_val[features], y_val, weight=val_weights)
#         model = lgb.train(params=params,
#                           num_boost_round=1400,
#                           train_set=train_dataset,
#                           valid_sets=[train_dataset, val_dataset],
#                           verbose_eval=250,
#                           early_stopping_rounds=50,
#                           feval=feval_mae)
#         # Add predictions to the out of folds array
#         oof_predictions[val_ind] = model.predict(x_val[features])
#         # Predict the test set
#         test_predictions += model.predict(test[features]) / 5
#     mae_score = mean_absolute_error(y, oof_predictions)
#     print(f'Our out of folds MAE is {mae_score}')
#
#     print('Plotting feature importances...')
#     ax = lgb.plot_importance(model, max_num_features=20)
#     plt.savefig('feature_importance.png')
#
#     # Return test predictions
#     return test_predictions


def train_and_evaluate_lgb(train, test):
    print('data prepare')
    features = [col for col in train.columns if col not in {'id', 'pressure', 'breath_id'}]
    y = train['pressure']
    oof_predictions = np.zeros(train.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(test.shape[0])
    print('Create a KFold object')
    kfold = KFold(n_splits=5, random_state=SEED, shuffle=True)
    gkfold = GroupKFold(n_splits=5)
    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(gkfold.split(train, y, groups=train['breath_id'])):
        print(f'Training fold {fold + 1}')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train[features], y_train, weight=train_weights)
        val_dataset = lgb.Dataset(x_val[features], y_val, weight=val_weights)
        model = lgb.train(params=PARAMS,
                          num_boost_round=1600,
                          train_set=train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          verbose_eval=50,
                          early_stopping_rounds=20,
                          feval=feval_mae)
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val[features])
        # Predict the test set
        test_predictions += model.predict(test[features]) / 5
    mae_score = mean_absolute_error(y, oof_predictions)
    print(f'Our out of folds MAE is {mae_score}')

    print('Plotting feature importances...')
    ax = lgb.plot_importance(model, max_num_features=20)
    plt.savefig('feature_importance.png')

    # Return test predictions
    return test_predictions



