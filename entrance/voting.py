import pickle
import warnings
from pathlib import Path
from time import time

import pandas as pd
import numpy as np

from util.jsons import of_json, to_json
from constant import *

warnings.filterwarnings("ignore")


def calculate_score(dfs, weights=(1, 1, 1)):
    assert len(dfs) == len(weights)
    result = pd.DataFrame(columns=['id', 'predicts'])
    result['id'] = dfs[0]['id']
    result['predicts'] = 0
    for i in range(len(dfs)):
        result['predicts'] = result['predicts'] + dfs[i]['predicts'] * (weights[i] / sum(weights))
    return result


def vote():
    lgb_result = pd.read_csv(Path(dir_result).joinpath('oneday_lgbm_model_1_submission.csv'), encoding='utf-8')
    xgb_result = pd.read_csv(Path(dir_result).joinpath('oneday_xgb_model_1_submission.csv'), encoding='utf-8')
    cat_result = pd.read_csv(Path(dir_result).joinpath('oneday_catm_model_1_submission.csv'), encoding='utf-8')

    result = calculate_score([lgb_result, xgb_result, cat_result])
    result.to_csv(Path(dir_result) / 'oneday_voting_1_submission.csv', index=False)


vote()
