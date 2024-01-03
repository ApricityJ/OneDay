import pickle
from pathlib import Path
import warnings
from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pylab as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (fbeta_score, precision_score, recall_score, confusion_matrix, classification_report)
from sklearn.utils import Bunch
import shap

from util.metrics import (lgb_f2_score_eval, get_best_f2_threshold, focal_loss_lgb_1, focal_loss_lgb,
                          lgb_f1_score_multi_macro_eval, lgb_f1_score_multi_weighted_eval)
from util.hyperopt import Hyperopt
from util.optuna import Optuna
from util.jsons import to_json

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_rows', None)


class LightGBM(object):
    def __init__(self, dataset: str,
                       train_set: list[pd.DataFrame],
                       predict_set: list[pd.DataFrame],
                       col_names: list[str],
                       category_cols: list[str],
                       objective: str,
                       metric: str | None,
                       num_class: int = 2,
                       optimizer: str = 'hyperopt',
                       magic_seed: int = 29,
                       out_dir: Path = Path('result'),
                       out_model_name: str = 'result_model_lgb.p',
                       save: bool = False,
                       version: str = '1',
                       n_folds: int = 5,
                       fobj: Optional[Callable] = None,
                       feval: Optional[Callable] = None,
                       eval_key: str = None,
                       hyperopt_max_evals: int = 30,
                       optuna_n_trials: int = 20,
                       optuna_direction: str = 'maximize'
                 ):

        self.dataset = dataset
        self.col_names = col_names
        self.category_cols = category_cols

        self.X_tr, self.y_tr = train_set[0], train_set[1]
        self.lgb_train = lgb.Dataset(
            self.X_tr, self.y_tr,
            feature_name=self.col_names,
            categorical_feature=self.category_cols,
            free_raw_data=False)
        # self.lgb_valid = self.lgb_train.create_valid(self.X_val, self.y_val)
        self.X_predict = predict_set[0]
        self.id_predict = predict_set[1]

        self.objective = objective
        self.num_class = num_class
        self.optimizer = optimizer
        self.magic_seed = magic_seed

        self.out_dir = out_dir
        self.out_model_name = out_model_name
        self.save = save
        self.version = version

        self.n_folds = n_folds
        self.fobj = fobj
        self.feval = feval
        self.eval_key = eval_key
        self.metric = metric

        self.hyperopt_max_evals = hyperopt_max_evals
        self.optuna_n_trials = optuna_n_trials
        self.optuna_direction = optuna_direction


    def optimize(self) -> dict:
        if self.optimizer == "hyperopt":
            optimizer_ = Hyperopt("lightgbm", self)
        elif self.optimizer == "optuna":
            optimizer_ = Optuna("lightgbm", self)
        else:
            optimizer_ = None
            pass
        return optimizer_.optimize()


    def train_and_predict(self, params):
        print("--------- begin training and predicting ---------")

        params['objective'] = self.objective
        params['num_class'] = self.num_class
        params['metric'] = self.metric
        params['verbose'] = -1

        if self.objective == 'multiclass':
            eval_prediction_folds = dict()
            prediction_folds_mean = np.zeros((self.X_predict.shape[0], self.num_class))
            score_folds = {"f1-macro": 0, "f1-weighted": 0}  # TODO: fix
        else:
            eval_prediction_folds = pd.DataFrame()
            prediction_folds_mean = np.zeros(len(self.X_predict))
            score_folds = []

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.magic_seed, shuffle=True)
        for index, (train_index, eval_index) in enumerate(kf.split(self.X_tr, self.y_tr)):
            print(f"FOLD : {index}")
            train_part = lgb.Dataset(self.X_tr.loc[train_index],
                                     self.y_tr.loc[train_index],
                                     feature_name=self.col_names,
                                     categorical_feature=self.category_cols)

            eval_part = lgb.Dataset(self.X_tr.loc[eval_index],
                                    self.y_tr.loc[eval_index],
                                    feature_name=self.col_names,
                                    categorical_feature=self.category_cols)

            model = lgb.train(params,
                              train_part,
                              fobj=self.fobj,
                              feval=self.feval,
                              valid_sets=[train_part, eval_part],
                              valid_names=['train', 'valid'],
                              verbose_eval=1)

            prediction_folds_mean += (model.predict(self.X_predict) / self.n_folds)
            eval_prediction = model.predict(self.X_tr.loc[eval_index])

            if self.objective == 'multiclass':
                for item_index, item in zip(eval_index, eval_prediction):
                    eval_prediction_folds[int(item_index)] = list(item)

                eval_prediction = np.argmax(eval_prediction, axis=1)
                f1_macro = f1_score(self.y_tr.loc[eval_index], eval_prediction, average="macro")
                f1_weighted = f1_score(self.y_tr.loc[eval_index], eval_prediction, average="weighted")
                print(f"FOLD f1-macro: {f1_macro}, f1-weighted: {f1_weighted}")
                score_folds['f1-macro'] += (f1_macro / self.n_folds)
                score_folds['f1-weighted'] += (f1_weighted / self.n_folds)
            else:
                eval_df = pd.DataFrame({'id': eval_index, 'predicts': eval_prediction})
                if index == 0:
                    eval_prediction_folds = eval_df.copy()
                else:
                    eval_prediction_folds = eval_prediction_folds.append(eval_df)

                best_f2, best_threshold = get_best_f2_threshold(eval_prediction, self.y_tr.loc[eval_index])
                score_folds.append(best_f2)
                print(f"FOLD F2 = {best_f2}")

        # binary
        print(f'score all : {score_folds}')
        print(f'score mean : {sum(score_folds) / self.n_folds}')
        self._validate_and_predict_binary(eval_prediction_folds, prediction_folds_mean, params)
        print("--------- done training and predicting ---------")

        # multiclass
        print(f'score mean: \n{score_folds}')
        self._validate_and_predict_multiclass(eval_prediction_folds, prediction_folds_mean, params)
        print("--------- done training and predicting ---------")


    def _validate_and_predict_binary(self, eval_prediction_folds, prediction_folds_mean, params):

        eval_predictions = eval_prediction_folds.sort_values(by=['id'])
        # print(eval_predictions.head())
        # print(eval_predictions.shape)
        best_f2, best_threshold = get_best_f2_threshold(eval_predictions['predicts'].values, self.y_tr)
        diff_threshold = np.quantile(eval_predictions['predicts'].values, 0.8) - best_threshold
        print(f'best F2-Score : {best_f2}')
        print(f"quantile 80% train : {np.quantile(eval_predictions['predicts'].values, 0.8)}")
        print(f'best threshold : {best_threshold}')
        print(f'diff between quantile and threshold : {diff_threshold}')

        eval_predictions_classify = (eval_predictions['predicts'].values > best_threshold).astype('int')
        # acc = accuracy_score(self.y_tr, eval_predictions)
        f2 = fbeta_score(self.y_tr, eval_predictions_classify, beta = 2)
        precision = precision_score(self.y_tr, eval_predictions_classify)
        recall = recall_score(self.y_tr, eval_predictions_classify)
        cm = confusion_matrix(self.y_tr, eval_predictions_classify)
        print('F2-Score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(f2, precision, recall))
        print('confusion_matrix:')
        print(cm)

        # print(prediction_folds_mean)
        # print(prediction_folds_mean.shape)
        print(f"quantile 80% test : {np.quantile(prediction_folds_mean, 0.8)}")
        test_threshold = np.quantile(prediction_folds_mean, 0.8) - diff_threshold
        print(f'test threshold : {test_threshold}')

        eval_predictions.to_csv(self.out_dir / '{}_lgbm_model_{}_train.csv'.format(self.dataset, self.version),
                                index=False)
        pd.DataFrame({'id': self.id_predict, 'predicts': prediction_folds_mean})\
            .to_csv(self.out_dir / '{}_lgbm_model_{}_test.csv'.format(self.dataset, self.version), index=False)

        submission = pd.DataFrame({'id': self.id_predict,
                                   'predicts': np.where(prediction_folds_mean > test_threshold, 1, 0)})
        submission['predicts'] = submission['predicts'].astype(int)
        submission.to_csv(self.out_dir / '{}_lgbm_model_{}_submission.csv'.format(self.dataset, self.version),
                          index=False)

        if self.save:
            params['verbose'] = -1
            print('train and save model with all data.')
            model = lgb.train(params, self.lgb_train, fobj=self.fobj)
            results = Bunch(f2=f2, precision=precision, recall=recall, cm=cm, test_threshold=test_threshold)
            results.model = model
            results.best_params = params
            results.columns = self.col_names
            pickle.dump(results, open(self.out_dir / self.out_model_name, 'wb'))


    def _validate_and_predict_multiclass(self, eval_prediction_folds, prediction_folds_mean, params):

        eval_prediction_folds = dict(sorted(eval_prediction_folds.items(), key=lambda item: item[0]))
        eval_prediction = list(eval_prediction_folds.values())
        eval_prediction = np.argmax(eval_prediction, axis=1)

        target_names = ['class ' + str(i) for i in range(self.num_class)]
        print(classification_report(self.y_tr, eval_prediction, target_names=target_names))

        to_json(eval_prediction_folds, self.out_dir / '{}_lgbm_model_{}_train.json'.format(self.dataset, self.version))
        test_prediction = {k: list(v) for k, v in enumerate(prediction_folds_mean)}
        to_json(test_prediction, self.out_dir / '{}_lgbm_model_{}_test.json'.format(self.dataset, self.version))

        submission_prediction = np.argmax(prediction_folds_mean, axis=1)
        submission = pd.DataFrame({'id': self.id_predict, 'predicts': submission_prediction})
        # submission['predicts'] = submission['predicts'].astype(int)
        submission.to_csv(self.out_dir / '{}_lgbm_model_{}_submission.csv'.format(self.dataset, self.version),
                          index=False)

        if self.save:
            params['verbose'] = -1
            print('train and save model with all data.')
            model = lgb.train(params, self.lgb_train, fobj=self.fobj)
            results = Bunch(model=model, params=params, columns=self.col_names)
            pickle.dump(results, open(self.out_dir / self.out_model_name, 'wb'))

    @staticmethod
    def print_feature_importance(data_bunch):
        print(pd.DataFrame({
            'column': data_bunch.columns,
            'importance': data_bunch.model.feature_importance(),
        }).sort_values(by='importance', ascending=False))

        # plt.figure(figsize=(12, 6))
        # lgb.plot_importance(data_bunch.model, max_num_features=30)
        # plt.title("Feature Importance")
        # plt.show()

    @staticmethod
    def shap_feature_importance(data_bunch, X):
        shap_values = shap.TreeExplainer(data_bunch.model).shap_values(X)
        shap.summary_plot(shap_values, X)
        shap.summary_plot(shap_values[1], X)
