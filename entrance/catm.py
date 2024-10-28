import pickle
import warnings
from pathlib import Path
from time import time

import pandas as pd
import numpy as np
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

from model.CatBoost import CatBoost
from util.jsons import of_json, to_json
from constant import *
from util.hyperopt import Hyperopt
from util.optuna import Optuna
from util.metrics import (CatKSEvalMetric, lgb_f2_score_eval, get_best_f2_threshold)

warnings.filterwarnings("ignore")


def load_data(dir_path: Path, file_name: str) -> Bunch:
    data = pickle.load(open(dir_path.joinpath(file_name), 'rb'))
    return data


if __name__ == '__main__':
    args = {
        'dataset': 'xw_ent',
        'version': '1',
        'objective': 'Logloss',  # CrossEntropy...
        'num_class': 1,
        'optimizer': 'hyperopt',  # hyperopt, optuna...
        'save_experiment': True,
        'train_path': Path(dir_train),
        'test_path': Path(dir_test),
        'result_path': Path(dir_result),
        'train_file_name': file_name_train,
        'test_file_name': file_name_test,
        'out_model_name': 'result_model_catm.p',
        'magic_seed': active_random_state,
        'load_best_params': False,
        'params_file_name': 'best_params_catm.dict',
        'n_folds': 5,
        'target': 'train',

        'fobj': None,  # catboost用不到
        'feval': None,  # catboost用不到
        'eval_metric': 'CatKSEvalMetric',  # 'AUC'
        'eval_key': "test-CatKSEvalMetric-mean",  # 用于优化器  "test-AUC-mean"

        'hyperopt_max_evals': 30,  # 30
        'optuna_n_trials': 20,  # 20
        'optuna_direction': 'maximize'
    }
    print("-----------------------------")
    print(f"args : {args}")

    # start_time = time.time()

    # select_feature_and_prepare_data('flatmap')
    # create_data_bunch_from_csv()

    if 'multi' in args['objective']:
        assert args['num_class'] > 2, 'multiclass objective should have class num > 2.'
    assert args['params_file_name'] != '', 'please name the best params file.'

    if args['eval_metric'] is None:
        assert args['feval'] is not None and args['eval_key'] is not None, \
            "custom metric should be assigned when metric is None."

    if args['target'] == 'train':
        print("--------- begin load train and predict data ---------")
        train_data = load_data(args['train_path'], args['train_file_name'])
        print(f"columns : {train_data.col_names}")
        print(f"category columns : {train_data.category_cols}")
        X = train_data.data
        y = train_data.target
        print(f"X train shape : {X.shape}")
        print(f"y train shape : {y.shape}")

        test_data = load_data(args['test_path'], args['test_file_name'])
        X_predict = test_data.data
        id_predict = test_data.id
        print(f"X predict shape : {X_predict.shape}")
        print(f"id predict shape : {id_predict.shape}")
        print("--------- done load train and predict data ---------")

        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=1, stratify=y)
        start_time = time()

        model = CatBoost(
            dataset=args['dataset'],
            train_set=[X, y],
            predict_set=[X_predict, id_predict],
            col_names=train_data.col_names,
            category_cols=train_data.category_cols,
            objective=args['objective'],
            eval_metric=args['eval_metric'],
            num_class=args['num_class'],
            optimizer=args['optimizer'],
            magic_seed=args['magic_seed'],
            out_dir=args['result_path'],
            out_model_name=args['out_model_name'],
            save=args['save_experiment'],
            version=args['version'],
            n_folds=args['n_folds'],
            fobj=args['fobj'],
            feval=args['feval'],
            eval_key=args['eval_key'],
            hyperopt_max_evals=args['hyperopt_max_evals'],
            optuna_n_trials=args['optuna_n_trials'],
            optuna_direction=args['optuna_direction']
        )

        if args['load_best_params']:
            best_params = of_json(args['result_path'].joinpath(args['params_file_name']))
        else:
            best_params = model.optimize()
            best_params['depth'] = int(best_params['depth'])
            best_params['border_count'] = int(best_params['border_count'])
            best_params['num_boost_round'] = int(best_params['num_boost_round'])
            print(best_params)
            to_json(best_params, args['result_path'].joinpath(args['params_file_name']))

        print("--------- best params ---------")
        print(best_params)

        model.train_and_predict(best_params)

        # 打印特征重要性
        # assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'
        # data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
        # LightGBM.print_feature_importance(data_bunch)

        end_time = time()
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
        test_prediction = model.predict_proba(X_predict)[:, 1]
        print(test_prediction)

        if args['objective'] == 'Logloss':
            test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
            test_result.to_csv(args['result_path'] / '{}_catm_model_{}_test_from_all_data_model.csv'
                               .format(args['dataset'], args['version']), index=False)
        elif args['objective'] == 'multiclass':
            test_prediction = np.argmax(test_prediction, axis=1)
            test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
            test_result.to_csv(args['result_path'] / '{}_catm_model_{}_test_from_all_data_model.csv'
                               .format(args['dataset'], args['version']), index=False)
        else:
            pass

    elif args['target'] == 'feature_importance':
        assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'

        data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
        CatBoost.print_feature_importance(data_bunch)

        train_data_bunch = pickle.load(open(args['train_path'] / args['train_file_name'], 'rb'))
        X = train_data_bunch.data
        CatBoost.shap_feature_importance(data_bunch, X)
    else:
        pass
