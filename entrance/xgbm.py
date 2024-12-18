import pickle
import warnings
from pathlib import Path
import time

import pandas as pd
import numpy as np
import xgboost as xgb

from util.jsons import of_json, to_json
from model.XGBoost import XGBoost
from constant import *

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    args = {
        'dataset': 'xw_ent',  # penguins
        'version': '1',
        'objective': 'binary:logistic',  # binary:logistic, multi:softmax, multi:softprob...
        'eval_metric': None,  # ['logloss', 'auc'], ['mlogloss']
        'num_class': 2,
        'target': 'train',  # train, predict, feature_importance
        'optimizer': 'hyperopt',  # hyperopt, optuna...
        'save_experiment': True,
        # 'data_path': Path("../data"),
        'train_path': Path(dir_train),
        'test_path': Path(dir_test),
        'result_path': Path(dir_result),
        'train_file_name': file_name_train,
        'test_file_name': file_name_test,
        'out_model_name': 'result_model_xgb.p',
        'magic_seed': active_random_state,
        'load_best_params': False,
        'params_file_name': 'best_params_xgb.dict',

        'hyperopt_max_evals': 30,  # 30
        'optuna_n_trials': 20,  # 20
        'optuna_direction': 'maximize'
    }
    print("-----------------------------")
    print(f"args : {args}")

    start_time = time.time()

    if 'multi' in args['objective']:
        assert args['num_class'] > 2, 'multiclass objective should have class num > 2.'
    assert args['params_file_name'] != '', 'please name the best params file.'

    if args['target'] == 'train':
        print("--------- begin load train and predict data ---------")
        data_bunch = pickle.load(open(args['train_path'] / args['train_file_name'], 'rb'))
        col_names = data_bunch.col_names
        print(f"columns : {col_names}")
        category_cols = data_bunch.category_cols
        print(f"category columns : {category_cols}")
        X = data_bunch.data
        y = data_bunch.target
        print(f"X train shape : {X.shape}")
        print(f"y train shape : {y.shape}")

        data_bunch = pickle.load(open(args['test_path'] / args['test_file_name'], 'rb'))
        X_predict = data_bunch.data
        id_predict = data_bunch.id
        print(f"X predict shape : {X_predict.shape}")
        print(f"id predict shape : {id_predict.shape}")
        print("--------- done load train and predict data ---------")

        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=1, stratify=y)
        print("--------- begin test ---------")
        # if X.isnull().values.any():
        #     print('some null values in X')
        X.replace([np.inf, -np.inf], 1, inplace=True)
        X_predict.replace([np.inf, -np.inf], 1, inplace=True)
        # X.fillna(-1, inplace=True)
        # if not X.isnull().values.any():
        #     print('test again, no null values in X')
        # for name in col_names:
        #     print(name)
        #     test = xgb.DMatrix(X[name], y)
        print("---------- end test ----------")

        xgboost = XGBoost(
            dataset=args['dataset'],
            train_set=[X, y],
            predict_set=[X_predict, id_predict],
            col_names=col_names,
            objective=args['objective'],
            eval_metric=args['eval_metric'],
            num_class=args['num_class'],
            optimizer=args['optimizer'],
            magic_seed=args['magic_seed'],
            out_dir=args['result_path'],
            out_model_name=args['out_model_name'],
            save=args['save_experiment'],
            version=args['version'],
            hyperopt_max_evals=args['hyperopt_max_evals'],
            optuna_n_trials=args['optuna_n_trials'],
            optuna_direction=args['optuna_direction']
        )

        if args['load_best_params']:
            best_params = of_json(args['result_path'].joinpath(args['params_file_name']))
        else:
            best_params = xgboost.optimize()
            best_params['max_depth'] = int(best_params['max_depth'])
            to_json(best_params, args['result_path'].joinpath(args['params_file_name']))

        print("--------- best params ---------")
        print(best_params)

        if args['objective'] == 'binary:logistic':
            xgboost.train_and_predict_binary(best_params)
        elif args['objective'] == 'multi:softprob':
            xgboost.train_and_predict_multiclass(best_params)
        else:
            pass

        end_time = time.time()
        print(f'run time all : {(end_time - start_time) // 60} minutes.')

    elif args['target'] == 'predict':
        assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'

        print("--------- begin load predict data ---------")
        data_bunch = pickle.load(open(args['test_path'] / args['test_file_name'], 'rb'))
        X_predict = data_bunch.data
        id_predict = data_bunch.id
        print(f"X predict shape : {X_predict.shape}")
        print(f"id predict shape : {id_predict.shape}")
        print("--------- done load predict data ---------")

        data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
        model = data_bunch.model
        test_prediction = model.predict(xgb.DMatrix(X_predict))
        print(test_prediction)

        if args['objective'] == 'binary:logistic':
            test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
            test_result.to_csv(args['result_path'] / '{}_xgb_model_{}_test_from_all_data_model.csv'
                               .format(args['dataset'], args['version']), index=False)
        elif args['objective'] == 'multi:softprob':
            test_prediction = np.argmax(test_prediction, axis=1)
            test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
            test_result.to_csv(args['result_path'] / '{}_xgb_model_{}_test_from_all_data_model.csv'
                               .format(args['dataset'], args['version']), index=False)
        else:
            pass

    elif args['target'] == 'feature_importance':
        assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'

        data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
        # XGBoost.print_feature_importance(data_bunch)

        train_data_bunch = pickle.load(open(args['train_path'] / args['train_file_name'], 'rb'))
        X = train_data_bunch.data
        XGBoost.shap_feature_importance(data_bunch, X)
    else:
        pass
