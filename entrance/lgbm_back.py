# import pickle
# import warnings
# from pathlib import Path
# from time import time
#
# import pandas as pd
# import numpy as np
# from sklearn.utils import Bunch
# from sklearn.model_selection import train_test_split
#
# from model.lightGBM import LightGBM
# from util.jsons import of_json, to_json
# from constant import *
# from util.hyperopt import Hyperopt
# from util.optuna import Optuna
# from util.metrics import (lgb_f2_score_eval, get_best_f2_threshold, focal_loss_lgb_1, focal_loss_lgb,
#                           lgb_f1_score_multi_macro_eval, lgb_f1_score_multi_weighted_eval)
#
# warnings.filterwarnings("ignore")
#
#
# default_params_dict = {
#         'dataset': 'oneday',
#         'version': '1',
#         'objective': 'binary',  # binary, multiclass...
#         'metric': 'auc',  # None需在模型中指定
#         'num_class': 2,
#         'optimizer': 'hyperopt',  # hyperopt, optuna...
#         'save_experiment': True,
#         'train_path': Path(dir_train),
#         'test_path': Path(dir_test),
#         'result_path': Path(dir_result),
#         'train_file_name': file_name_train,
#         'test_file_name': file_name_test,
#         'out_model_name': 'result_model_lgbm.p',
#         'magic_seed': active_random_state,
#         'load_best_params': False,
#         'params_file_name': 'best_params_lgbm.dict',
#         'n_folds': 5,
#
#         'fobj': lambda x, y: focal_loss_lgb(x, y, alpha=0.25, gamma=2.0),  # 默认None
#         #lambda x, y: f1_score_multi_macro_eval(x, y, self.num_class)
#         # self.eval_key = "f1-macro-mean"
#         'feval': 'lgb_f2_score_eval',  # 默认None
#         'eval_key': "f2-mean",  # 用于优化器
#
#         'hyperopt_max_evals': 30,
#         'optuna_n_trials': 20,
#         'optuna_direction': 'maximize'
# }
#
#
# def load_data(dir_path: Path, file_name: str) -> Bunch:
#     data = pickle.load(open(dir_path.joinpath(file_name), 'rb'))
#     return data
#
#
# def timer(func):
#     def func_wrapper(*args, **kwargs):
#         time_start = time()
#         result = func(*args, **kwargs)
#         time_end = time()
#         time_spend = time_end - time_start
#         print(f'{func.__name__} cost time {time_spend // 60} minutes.')
#         return result
#     return func_wrapper
#
#
# class LightGBMEntrance(object):
#     def __init__(self, params_dict=None):
#
#         if params_dict is None:
#             params_dict = default_params_dict
#         self.dataset = params_dict['dataset']
#         self.version = params_dict['version']
#         self.objective = params_dict['objective']
#         self.metric = params_dict['metric']
#         self.num_class = params_dict['num_class']
#         self.optimizer = params_dict['optimizer']
#         self.save_experiment = params_dict['save_experiment']
#         self.train_path = params_dict['train_path']
#         self.test_path = params_dict['test_path']
#         self.result_path = params_dict['result_path']
#         self.train_file_name = params_dict['train_file_name']
#         self.test_file_name = params_dict['test_file_name']
#         self.out_model_name = params_dict['out_model_name']
#         self.magic_seed = params_dict['magic_seed']
#         self.load_best_params = params_dict['load_best_params']
#         self.params_file_name = params_dict['params_file_name']
#         self.n_folds = params_dict['n_folds']
#         self.fobj = params_dict['fobj']
#         self.feval = params_dict['feval']
#         self.eval_key = params_dict['eval_key']
#         self.hyperopt_max_evals = params_dict['hyperopt_max_evals']
#         self.optuna_n_trials = params_dict['optuna_n_trials']
#         self.optuna_direction = params_dict['optuna_direction']
#
#         self.model = None
#         self.best_params = None
#
#         if self.objective == 'multiclass':
#             assert self.num_class > 2, 'multiclass objective should have class num > 2.'
#         assert self.params_file_name != '', 'please name the best params file.'
#
#         if self.metric is None:
#             assert self.feval is not None and self.eval_key is not None, \
#                 "custom metric should be assigned when metric is None."
#
#
#     def __str__(self):
#         return '\n'.join(f'{item[0]}: {item[1]}' for item in self.__dict__.items())
#
#     @timer
#     def train(self):
#         print("--------- begin load train and predict data ---------")
#         train_data = load_data(self.train_path, self.train_file_name)
#         print(f"columns : {train_data.col_names}")
#         print(f"category columns : {train_data.category_cols}")
#         X = train_data.data
#         y = train_data.target
#         print(f"X train shape : {X.shape}")
#         print(f"y train shape : {y.shape}")
#
#         test_data = load_data(self.test_path, self.test_file_name)
#         X_predict = test_data.data
#         id_predict = test_data.id
#         print(f"X predict shape : {X_predict.shape}")
#         print(f"id predict shape : {id_predict.shape}")
#         print("--------- done load train and predict data ---------")
#
#         # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=1, stratify=y)
#
#         self.model = LightGBM(
#             dataset = self.dataset,
#             train_set = [X, y],
#             predict_set = [X_predict, id_predict],
#             col_names = train_data.col_names,
#             category_cols = train_data.category_cols,
#             objective = self.objective,
#             metric = self.metric,
#             num_class = self.num_class,
#             optimizer = self.optimizer,
#             magic_seed = self.magic_seed,
#             out_dir = self.result_path,
#             out_model_name = self.out_model_name,
#             save = self.save_experiment,
#             version = self.version,
#             n_folds = self.n_folds,
#             fobj = self.fobj,
#             feval = self.feval,
#             eval_key = self.eval_key,
#             hyperopt_max_evals = self.hyperopt_max_evals,
#             optuna_n_trials = self.optuna_n_trials,
#             optuna_direction = self.optuna_direction
#         )
#
#         if self.load_best_params:
#             self.best_params = of_json(self.result_path.joinpath(self.params_file_name))
#         else:
#             self.best_params = self.model.optimize()
#             self.best_params['num_leaves'] = int(self.best_params['num_leaves'])
#             to_json(self.best_params, self.result_path.joinpath(self.params_file_name))
#
#         print("--------- best params ---------")
#         print(self.best_params)
#
#         self.model.train_and_predict(self.best_params)
#
#         # 打印特征重要性
#         # assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'
#         # data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
#         # LightGBM.print_feature_importance(data_bunch)
#
#
#
#
# if __name__ == '__main__':
#
#
#     elif args['target'] == 'predict':
#         assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'
#
#         print("--------- begin load predict data ---------")
#         data_bunch = pickle.load(open(args['test_path'] / args['test_file_name'], 'rb'))
#         X_predict = data_bunch.data
#         id_predict = data_bunch.id
#         print(f"X predict shape : {X_predict.shape}")
#         print(f"id predict shape : {id_predict.shape}")
#         print("--------- done load predict data ---------")
#
#         data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
#         model = data_bunch.model
#         test_prediction = model.predict(X_predict)
#         print(test_prediction)
#
#         if args['objective'] == 'binary':
#             test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
#             test_result.to_csv(args['result_path'] / '{}_lgbm_model_{}_test_from_all_data_model.csv'
#                                .format(args['dataset'], args['version']), index=False)
#         elif args['objective'] == 'multiclass':
#             test_prediction = np.argmax(test_prediction, axis=1)
#             test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
#             test_result.to_csv(args['result_path'] / '{}_lgbm_model_{}_test_from_all_data_model.csv'
#                                .format(args['dataset'], args['version']), index=False)
#         else:
#             pass
#
#     elif args['target'] == 'feature_importance':
#         assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'
#
#         data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
#         LightGBM.print_feature_importance(data_bunch)
#
#         # train_data_bunch = pickle.load(open(args['train_path'] / args['train_file_name'], 'rb'))
#         # X = train_data_bunch.data
#         # LightGBM.shap_feature_importance(data_bunch, X)
#     else:
#         pass
