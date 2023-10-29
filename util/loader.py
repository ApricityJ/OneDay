from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
from numpy import datetime64
from pandas import DataFrame

from constant import *
# from util.logger import logger



def shrink_df(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        # logger.info('Memory usage shrink to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
        #         start_mem - end_mem) / start_mem))
        pass
    return df.round(3)


def to_df(csv: Path):
    return pd.read_csv(csv, encoding='utf-8')


def to_ds(p: Path):
    ds: Dict[str, DataFrame] = {}
    csv_list = p.glob('**/*.csv')
    for csv in csv_list:
        k = csv.stem.replace('.csv', '')
        v = to_df(csv.as_uri())
        ds[k] = shrink_df(v)
    return ds


def to_ds_train():
    return to_ds(dir_train)


def to_ds_test():
    return to_ds(dir_test)


def to_ds_preprocess():
    return to_ds(dir_preprocess)


def to_df_result():
    return to_df(Path(dir_result).joinpath('result.csv'))


# 包含train和test的数据集
def to_ds_train_test():
    ds: Dict[str, DataFrame] = {}
    for f in os.listdir(dir_train):
        key = f.replace('.csv', '')
        ds[key] = to_df_train_test(key)
    return ds


# train和test，如果输入key返回某一张表的pair
# e.g. XW_AGET_PAY.csv XW_AGET_PAY_A.csv
# 否则返回 默认的train/test
def to_df_train_test(key=None):
    if key:
        df_train = to_df(Path(dir_train).joinpath(f'{key}.csv'))
        df_test = to_df(Path(dir_test).joinpath(f'{key}_A.csv'))
    else:
        df_train = to_df(Path(dir_preprocess).joinpath('train.csv'))
        df_test = to_df(Path(dir_preprocess).joinpath('test.csv'))
    return [df_train, df_test]


# 图特征的原始数据
def to_df_graph():
    return to_df(Path(dir_preprocess).joinpath('graph.csv'))


def ls(p: Path):
    return os.listdir(p)


def to_concat_df(key=None) -> DataFrame:
    df_train, df_test = to_df_train_test(key)
    df_train['SRC'] = 'train'
    df_test['SRC'] = 'test'
    return pd.concat([df_train, df_test])