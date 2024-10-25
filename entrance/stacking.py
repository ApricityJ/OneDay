import pickle
import warnings
from pathlib import Path
from time import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from util.jsons import of_json, to_json
from constant import *
import util.metrics as metrics

warnings.filterwarnings("ignore")


def stack():
    lgb_train = pd.read_csv(Path(dir_result).joinpath('oneday_lgbm_model_1_train.csv'), encoding='utf-8')
    xgb_train = pd.read_csv(Path(dir_result).joinpath('oneday_xgb_model_1_train.csv'), encoding='utf-8')
    cat_train = pd.read_csv(Path(dir_result).joinpath('oneday_catm_model_1_train.csv'), encoding='utf-8')

    train_x = pd.DataFrame(columns=['lgb_fea', 'xgb_fea', 'cat_fea'])
    train_x['lgb_fea'] = lgb_train['predicts']
    train_x['xgb_fea'] = xgb_train['predicts']
    train_x['cat_fea'] = cat_train['predicts']
    train_y = lgb_train['label']

    lgb_test = pd.read_csv(Path(dir_result).joinpath('oneday_lgbm_model_1_submission.csv'), encoding='utf-8')
    xgb_test = pd.read_csv(Path(dir_result).joinpath('oneday_xgb_model_1_submission.csv'), encoding='utf-8')
    cat_test = pd.read_csv(Path(dir_result).joinpath('oneday_catm_model_1_submission.csv'), encoding='utf-8')

    test_x = pd.DataFrame(columns=['lgb_fea', 'xgb_fea', 'cat_fea'])
    test_x['lgb_fea'] = lgb_test['predicts']
    test_x['xgb_fea'] = xgb_test['predicts']
    test_x['cat_fea'] = cat_test['predicts']
    test_id = lgb_test['id']

    model = LogisticRegression()
    model.fit(train_x, train_y)
    result = model.predict_proba(test_x)[:, 1]

    result_train = model.predict_proba(train_x)[:, 1]
    auc_score = metrics.auc_score(train_y, result_train)
    print(auc_score)
    ks_score = metrics.CatKSEvalMetric_custom(result_train, train_y)
    print(ks_score)

    # todo: 把train的结果输出一下
    # pd.DataFrame({'id': test_id, 'predicts': result}) \
    #     .to_csv(Path(dir_result) / 'oneday_stacking_1_submission.csv', index=False)

    # pd.DataFrame({'id': test_id, 'predicts': result}) \
    #     .to_csv(Path(dir_result) / 'oneday_stacking_1_submission.csv', index=False)


stack()
