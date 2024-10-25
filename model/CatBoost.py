from __future__ import annotations

import pickle
from pathlib import Path
import warnings
import re
from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd
import catboost as cb
import matplotlib.pylab as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (fbeta_score, precision_score, recall_score, confusion_matrix, classification_report)
from sklearn.utils import Bunch
import shap

import util.metrics as metrics
from util.hyperopt import Hyperopt
from util.optuna import Optuna
from util.jsons import to_json

warnings.filterwarnings("ignore")
# plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_rows', None)


class CatBoost(object):
    def __init__(self, dataset: str,
                 train_set: list[pd.DataFrame],
                 predict_set: list[pd.DataFrame],
                 col_names: list[str],
                 category_cols: list[str],
                 objective: str,
                 eval_metric: str | None,
                 num_class: int = 2,
                 optimizer: str = 'hyperopt',
                 magic_seed: int = 29,
                 out_dir: Path = Path('result'),
                 out_model_name: str = 'result_model_lgb.p',
                 save: bool = False,
                 version: str = '1',
                 n_folds: int = 5,
                 fobj: Optional[Callable] = None,
                 feval: str = None,
                 eval_key: str = None,
                 hyperopt_max_evals: int = 30,
                 optuna_n_trials: int = 20,
                 optuna_direction: str = 'maximize'
                 ):

        self.dataset = dataset
        self.col_names = col_names
        self.category_cols = category_cols

        self.X_tr, self.y_tr = train_set[0], train_set[1]
        self.X_predict = predict_set[0]
        self.id_predict = predict_set[1]
        self.cb_train = cb.Pool(self.X_tr, self.y_tr, cat_features=self.category_cols)

        self.objective = objective
        self.num_class = num_class
        self.optimizer = optimizer
        self.magic_seed = magic_seed

        self.out_dir = out_dir
        self.out_model_name = out_model_name
        self.save = save
        self.version = version

        self.n_folds = n_folds
        self.fobj = None
        self.feval = None
        self.eval_metric_custom = None
        self.eval_metric = eval_metric
        if eval_metric not in ('AUC'):
            self.eval_metric = getattr(metrics, eval_metric)()
            self.eval_metric_custom = getattr(metrics, eval_metric + '_custom')
        self.eval_key = eval_key

        self.hyperopt_max_evals = hyperopt_max_evals
        self.optuna_n_trials = optuna_n_trials
        self.optuna_direction = optuna_direction

    def optimize(self) -> dict:
        if self.optimizer == "hyperopt":
            optimizer_ = Hyperopt("catboost", self)
        elif self.optimizer == "optuna":
            optimizer_ = Optuna("catboost", self)
        else:
            optimizer_ = None
            pass
        return optimizer_.optimize()

    def train_and_predict(self, params):
        print("--------- begin training and predicting ---------")

        params['objective'] = self.objective
        # params['iterations'] = params['num_boost_round']
        # params['fobj'] = self.fobj
        # params['num_class'] = self.num_class
        params['eval_metric'] = self.eval_metric
        params['verbose'] = False
        params['allow_writing_files'] = False

        if self.objective == 'multiclass':
            eval_prediction_folds = dict()
            prediction_folds_mean = np.zeros((self.X_predict.shape[0], self.num_class))
        else:
            eval_prediction_folds = pd.DataFrame()
            prediction_folds_mean = np.zeros(len(self.X_predict))

        score_folds = []
        threshold_folds = []
        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.magic_seed, shuffle=True)
        for index, (train_index, eval_index) in enumerate(kf.split(self.X_tr, self.y_tr)):
            print(f"FOLD : {index}")
            train_part = cb.Pool(self.X_tr.loc[train_index],
                                 self.y_tr.loc[train_index],
                                 cat_features=self.category_cols)

            eval_part = cb.Pool(self.X_tr.loc[eval_index],
                                self.y_tr.loc[eval_index],
                                cat_features=self.category_cols)

            model = cb.CatBoostClassifier(**params)
            model.fit(train_part,
                      eval_set=eval_part,
                      verbose_eval=False)

            prediction_folds_mean += (model.predict_proba(self.X_predict)[:, 1] / self.n_folds)
            eval_prediction = model.predict_proba(self.X_tr.loc[eval_index])[:, 1]

            if self.objective == 'multiclass':
                for item_index, item in zip(eval_index, eval_prediction):
                    eval_prediction_folds[int(item_index)] = list(item)
            else:
                eval_df = pd.DataFrame({'id': eval_index,
                                        'predicts': eval_prediction, 'label': self.y_tr.loc[eval_index]})
                if index == 0:
                    eval_prediction_folds = eval_df.copy()
                else:
                    eval_prediction_folds = eval_prediction_folds.append(eval_df)

            # score, threshold = self.feval_custom(eval_prediction, self.y_tr.loc[eval_index])
            if self.eval_metric == 'AUC':
                score = metrics.auc_score(self.y_tr.loc[eval_index], eval_prediction)
                score_folds.append(score)
                print(f"FOLD SCORE = {score}")
            else:  # 自定义eval_metric
                score, threshold = self.eval_metric_custom(eval_prediction, self.y_tr.loc[eval_index])
                threshold_folds.append(threshold)
                score_folds.append(score)
                print(f"FOLD SCORE = {score}, FOLD THRESHOLD = {threshold}")

        print(f'score all : {score_folds}')
        print(f'score mean : {sum(score_folds) / self.n_folds}')

        # self._validate_and_predict(eval_prediction_folds, prediction_folds_mean, params)

        eval_predictions = eval_prediction_folds.sort_values(by=['id'])
        eval_predictions.to_csv(self.out_dir / '{}_catm_model_{}_train.csv'.format(self.dataset, self.version),
                                index=False)

        # 这里只是提交了概率
        pd.DataFrame({'id': self.id_predict, 'predicts': prediction_folds_mean}) \
            .to_csv(self.out_dir / '{}_catm_model_{}_submission.csv'.format(self.dataset, self.version), index=False)

        if self.save:
            self._save(params, model)

        print("--------- done training and predicting ---------")

    # def _validate_and_predict(self, eval_prediction_folds, prediction_folds_mean, params):
    #     if self.objective == 'multiclass':
    #         self._validate_and_predict_multiclass(eval_prediction_folds, prediction_folds_mean)
    #     else:
    #         self._validate_and_predict_binary(eval_prediction_folds, prediction_folds_mean)
    #
    #     if self.save:
    #         self._save(params)

    def _save(self, params, model):
        params['verbose'] = False
        print('train and save model with all data.')
        model.fit(self.cb_train)
        results = Bunch(model=model, params=params, columns=self.col_names)
        pickle.dump(results, open(self.out_dir / self.out_model_name, 'wb'))

    @staticmethod
    def print_feature_importance(data_bunch):
        # print(pd.DataFrame({
        #     'column': data_bunch.columns,
        #     'importance': data_bunch.model.get_feature_importance(),
        # }).sort_values(by='importance', ascending=False))
        print(data_bunch.model.get_feature_importance(prettified=True))

        # plt.figure(figsize=(12, 6))
        # lgb.plot_importance(data_bunch.model, max_num_features=30)
        # plt.title("Feature Importance")
        # plt.show()

    @staticmethod
    def shap_feature_importance(data_bunch, X):
        # 创建SHAP解释器并计算SHAP值
        explainer = shap.TreeExplainer(data_bunch.model)
        shap_values = explainer.shap_values(X)
        # print(np.array(shap_values).shape)  # (37050, 190)

        # shap.summary_plot(shap_values, X, plot_type="bar")
        # shap.summary_plot(shap_values, X, show=False, max_display=20)
        # plt.savefig("../picture/shap_summary_plot3.png")

        # shap.dependence_plot("YAVER_DPSA_BAL", shap_values, X, interaction_index=None, show=False)
        # plt.savefig("../picture/shap_dependence_plot2.png")

        shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0, :], matplotlib=True)

        plt.close()
