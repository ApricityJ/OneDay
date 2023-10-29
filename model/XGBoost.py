import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pylab as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score, confusion_matrix, classification_report)
from sklearn.utils import Bunch

from util.metrics import (xgb_f1_score_eval, get_best_f1_threshold,
                          xgb_f1_score_multi_macro_eval, xgb_f1_score_multi_weighted_eval)
from util.hyperopt import Hyperopt
from util.optuna import Optuna
from util.util import to_json

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class XGBoost(object):

    def __init__(self, dataset, train_set, predict_set, col_names,
                 objective, eval_metric, num_class=2, optimizer='hyperopt', magic_seed=29,
                 out_dir=Path('result'), out_model_name='result_model_xgb.p', save=False, version='1'):

        self.dataset = dataset
        self.col_names = col_names

        self.X_tr, self.y_tr = train_set[0], train_set[1]
        self.xgb_train = xgb.DMatrix(
            self.X_tr, self.y_tr,
            feature_names=self.col_names)
        self.X_predict = predict_set[0]
        self.id_predict = predict_set[1]
        self.xgb_predict = xgb.DMatrix(self.X_predict)

        self.objective = objective
        self.num_class = num_class
        self.optimizer = optimizer
        self.magic_seed = magic_seed

        self.out_dir = out_dir
        self.out_model_name = out_model_name
        self.save = save
        self.version = version

        self.n_folds = 5
        # self.obj = lambda x, y: f1_loss(x, y)  # 默认None
        self.obj = None
        self.feval = xgb_f1_score_eval  # 默认None
        # self.feval = lambda x, y: xgb_f1_score_multi_macro_eval(x, y, self.num_class)
        # self.feval = None
        self.eval_key = "test-f1-mean"
        # self.eval_key = "test-f1-macro-mean"
        self.eval_maximize = True
        self.eval_metric = eval_metric
        if self.eval_metric is None:
            assert self.feval is not None and self.eval_key is not None, \
                "custom metric should be assigned when metric is None."

    def optimize(self) -> dict:

        if self.optimizer == "hyperopt":
            optimizer_ = Hyperopt("xgboost", self)
        elif self.optimizer == "optuna":
            optimizer_ = Optuna("xgboost", self)
        else:
            optimizer_ = None
            pass
        return optimizer_.optimize()

    def train_and_predict_binary(self, params):
        print("--------- begin training and predicting ---------")

        params['objective'] = self.objective
        params['eval_metric'] = self.eval_metric
        params['verbosity'] = 0

        eval_prediction_folds = pd.DataFrame()
        prediction_folds_mean = np.zeros(len(self.X_predict))
        score_folds = []

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.magic_seed, shuffle=True)
        for index, (train_index, eval_index) in enumerate(kf.split(self.X_tr, self.y_tr)):
            print(f"FOLD : {index}")
            train_part = xgb.DMatrix(self.X_tr.loc[train_index],
                                     self.y_tr.loc[train_index],
                                     feature_names=self.col_names)

            eval_part = xgb.DMatrix(self.X_tr.loc[eval_index],
                                    self.y_tr.loc[eval_index],
                                    feature_names=self.col_names)

            model = xgb.train(params,
                              train_part,
                              num_boost_round=params['num_boost_round'],
                              obj=self.obj,
                              feval=self.feval,
                              maximize=self.eval_maximize,
                              evals=[(train_part, 'train'), (eval_part,'valid')],
                              verbose_eval=1)

            prediction_folds_mean += (model.predict(self.xgb_predict) / self.n_folds)
            eval_prediction = model.predict(eval_part)
            eval_df = pd.DataFrame({'id': eval_index, 'predicts': eval_prediction})
            if index == 0:
                eval_prediction_folds = eval_df.copy()
            else:
                eval_prediction_folds = eval_prediction_folds.append(eval_df)

            best_f1, best_threshold = get_best_f1_threshold(eval_prediction, self.y_tr.loc[eval_index])
            score_folds.append(best_f1)
            print(f"FOLD F1 = {best_f1}")

        print(f'score all : {score_folds}')
        print(f'score mean : {sum(score_folds) / self.n_folds}')
        self._validate_and_predict_binary(eval_prediction_folds, prediction_folds_mean, params)
        print("--------- done training and predicting ---------")

    def _validate_and_predict_binary(self, eval_prediction_folds, prediction_folds_mean, params):

        eval_predictions = eval_prediction_folds.sort_values(by=['id'])
        # print(eval_predictions.head())
        # print(eval_predictions.shape)
        best_f1, best_threshold = get_best_f1_threshold(eval_predictions['predicts'].values, self.y_tr)
        diff_threshold = np.quantile(eval_predictions['predicts'].values, 0.8) - best_threshold
        print(f'best F1-Score : {best_f1}')
        print(f"quantile 80% train : {np.quantile(eval_predictions['predicts'].values, 0.8)}")
        print(f'best threshold : {best_threshold}')
        print(f'diff between quantile and threshold : {diff_threshold}')

        eval_predictions_classify = (eval_predictions['predicts'].values > best_threshold).astype('int')
        # acc = accuracy_score(self.y_tr, eval_predictions)
        f1 = f1_score(self.y_tr, eval_predictions_classify)
        precision = precision_score(self.y_tr, eval_predictions_classify)
        recall = recall_score(self.y_tr, eval_predictions_classify)
        cm = confusion_matrix(self.y_tr, eval_predictions_classify)
        print('F1-Score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(f1, precision, recall))
        print('confusion_matrix:')
        print(cm)

        # print(prediction_folds_mean)
        # print(prediction_folds_mean.shape)
        print(f"quantile 80% test : {np.quantile(prediction_folds_mean, 0.8)}")
        test_threshold = np.quantile(prediction_folds_mean, 0.8) - diff_threshold
        print(f'test threshold : {test_threshold}')

        eval_predictions.to_csv(self.out_dir / '{}_xgb_model_{}_train.csv'.format(self.dataset, self.version),
                                index=False)
        pd.DataFrame({'id': self.id_predict, 'predicts': prediction_folds_mean})\
            .to_csv(self.out_dir / '{}_xgb_model_{}_test.csv'.format(self.dataset, self.version), index=False)

        submission = pd.DataFrame({'id': self.id_predict,
                                   'predicts': np.where(prediction_folds_mean > test_threshold, 1, 0)})
        submission['predicts'] = submission['predicts'].astype(int)
        submission.to_csv(self.out_dir / '{}_xgb_model_{}_submission.csv'.format(self.dataset, self.version),
                          index=False)

        if self.save:
            params['verbosity'] = 0
            print('train and save model with all data.')
            model = xgb.train(params, self.xgb_train, obj=self.obj)
            results = Bunch(f1=f1, precision=precision, recall=recall, cm=cm, test_threshold=test_threshold)
            results.model = model
            results.best_params = params
            pickle.dump(results, open(self.out_dir / self.out_model_name, 'wb'))

    def train_and_predict_multiclass(self, params):
        print("--------- begin training and predicting ---------")

        params['objective'] = self.objective
        params['num_class'] = self.num_class
        params['eval_metric'] = self.eval_metric
        params['verbosity'] = 0

        eval_prediction_folds = dict()
        prediction_folds_mean = np.zeros((self.X_predict.shape[0], self.num_class))
        score_folds = {"f1-macro": 0, "f1-weighted": 0}

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.magic_seed, shuffle=True)
        for index, (train_index, eval_index) in enumerate(kf.split(self.X_tr, self.y_tr)):
            print(f"FOLD : {index}")
            train_part = xgb.DMatrix(self.X_tr.loc[train_index],
                                     self.y_tr.loc[train_index],
                                     feature_names=self.col_names)

            eval_part = xgb.DMatrix(self.X_tr.loc[eval_index],
                                    self.y_tr.loc[eval_index],
                                    feature_names=self.col_names)

            model = xgb.train(params,
                              train_part,
                              num_boost_round=params['num_boost_round'],
                              obj=self.obj,
                              feval=self.feval,
                              maximize=self.eval_maximize,
                              evals=[(train_part, 'train'), (eval_part, 'valid')],
                              verbose_eval=1)

            prediction_folds_mean += (model.predict(self.xgb_predict) / self.n_folds)
            eval_prediction = model.predict(eval_part)
            for item_index, item in zip(eval_index, eval_prediction):
                eval_prediction_folds[int(item_index)] = list(map(float, item))
            # print(eval_prediction_folds)

            eval_prediction = np.argmax(eval_prediction, axis=1)
            f1_macro = f1_score(self.y_tr.loc[eval_index], eval_prediction, average="macro")
            f1_weighted = f1_score(self.y_tr.loc[eval_index], eval_prediction, average="weighted")
            print(f"FOLD f1-macro: {f1_macro}, f1-weighted: {f1_weighted}")
            score_folds['f1-macro'] += (f1_macro / self.n_folds)
            score_folds['f1-weighted'] += (f1_weighted / self.n_folds)

        print(f'score mean: \n{score_folds}')
        self._validate_and_predict_multiclass(eval_prediction_folds, prediction_folds_mean, params)
        print("--------- done training and predicting ---------")

    def _validate_and_predict_multiclass(self, eval_prediction_folds, prediction_folds_mean, params):

        eval_prediction_folds = dict(sorted(eval_prediction_folds.items(), key=lambda item: item[0]))
        eval_prediction = list(eval_prediction_folds.values())
        eval_prediction = np.argmax(eval_prediction, axis=1)

        target_names = ['class ' + str(i) for i in range(self.num_class)]
        print(classification_report(self.y_tr, eval_prediction, target_names=target_names))

        to_json(eval_prediction_folds, self.out_dir / '{}_xgb_model_{}_train.json'.format(self.dataset, self.version))
        test_prediction = {k: list(v) for k, v in enumerate(prediction_folds_mean)}
        to_json(test_prediction, self.out_dir / '{}_xgb_model_{}_test.json'.format(self.dataset, self.version))

        submission_prediction = np.argmax(prediction_folds_mean, axis=1)
        submission = pd.DataFrame({'id': self.id_predict, 'predicts': submission_prediction})
        # submission['predicts'] = submission['predicts'].astype(int)
        submission.to_csv(self.out_dir / '{}_xgb_model_{}_submission.csv'.format(self.dataset, self.version),
                          index=False)

        if self.save:
            params['verbosity'] = 0
            print('train and save model with all data.')
            model = xgb.train(params, self.xgb_train, obj=self.obj)
            results = Bunch(model=model, params=params)
            pickle.dump(results, open(self.out_dir / self.out_model_name, 'wb'))

    @staticmethod
    def print_feature_importance(data_bunch):
        importance_dict = data_bunch.model.get_fscore()
        print(pd.DataFrame({
            'column': importance_dict.keys(),
            'importance': importance_dict.values(),
        }).sort_values(by='importance', ascending=False))

        xgb.plot_importance(data_bunch.model, max_num_features=30)
        plt.title("Feature Importance")
        plt.show()

    @staticmethod
    def shap_feature_importance(data_bunch, X):
        shap_values = shap.TreeExplainer(data_bunch.model).shap_values(X)
        # shap.summary_plot(shap_values, X, plot_type="bar")
        shap.summary_plot(shap_values, X)
