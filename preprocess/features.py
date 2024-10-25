from typing import List
from inspect import isfunction
import operator
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from category_encoders.target_encoder import TargetEncoder
from category_encoders import OneHotEncoder, CatBoostEncoder

from constant import *
from data import loader, exporter


def cartesian_product_basic(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    返回笛卡尔积
    :param left: 原始数据集
    :param right: 扩展的列
    :return: 扩展后的数据集
    """
    return left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1)


def handle_outliers_by_quantile(data: pd.DataFrame, col_name: str, upper_percent: float, lower_percent: float,
                                is_train_name: str = 'SRC') -> pd.DataFrame:
    """
    覆盖过大或过小的值
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param upper_percent: 上限
    :param lower_percent: 下限
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()

    upper_lim = tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name].quantile(upper_percent)  # 这里只用训练数据
    lower_lim = tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name].quantile(lower_percent)
    tmp_data.loc[(tmp_data[col_name] > upper_lim), col_name] = upper_lim
    tmp_data.loc[(tmp_data[col_name] < lower_lim), col_name] = lower_lim
    return tmp_data[col_name]


def category_to_num(data: pd.DataFrame, col_name: str, is_train_name: str = 'SRC') -> pd.DataFrame:
    """
    类别变量简单转换为整数
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()
    col_type_dic = {label: idx for idx, label in
                    enumerate(np.unique(tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name]))}
    tmp_data[col_name] = tmp_data[col_name].map(col_type_dic)  # 这里没有考虑key不存在的情况，默认NaN，可以改成其他的
    return tmp_data[col_name].fillna(-1).astype(int)  # 为了转成int，这里填充了-1，可以改成其他的


def category_to_num_by_label_encoder(data: pd.DataFrame, col_name: str,
                                     is_train_name: str = 'SRC') -> pd.DataFrame:
    """
    使用sklearn LabelEncoder将类别变量转为数字
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name])
    test_tmp_data = tmp_data.loc[(tmp_data[is_train_name] == 'test'), col_name]
    test_tmp_data = test_tmp_data.map(lambda x: '<unknown>' if x not in encoder.classes_ else x)  # 处理test标签不在train中的情况
    tmp_data.loc[(tmp_data[is_train_name] == 'test'), col_name] = test_tmp_data
    encoder.classes_ = np.append(encoder.classes_, '<unknown>')
    return encoder.transform(tmp_data[col_name])


def category_encoding_by_onehot(data: pd.DataFrame, col_name: str, is_train_name: str = 'SRC') -> pd.DataFrame:
    """
    类别变量onehot化
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()
    encoder = OneHotEncoder(cols=[col_name], handle_unknown='indicator', handle_missing='indicator', use_cat_names=True)
    encoder.fit(tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name])
    return encoder.transform(tmp_data[col_name])


def category_encoding_by_target_encoder(data: pd.DataFrame, col_name: str, label_name: str = LABEL,
                                        is_train_name: str = 'SRC') -> pd.DataFrame:
    """
    类别变量target encoding处理
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param label_name: 标签类名
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, label_name, is_train_name]].copy()
    encoder = TargetEncoder(cols=[col_name], handle_unknown='value', handle_missing='value') \
        .fit(tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name],
             tmp_data.loc[(tmp_data[is_train_name] == 'train'), label_name])  # 在训练集上训练
    return encoder.transform(tmp_data[col_name])


# 交叉验证的目标编码
def category_encoding_by_cross_val_target_encoder(data: pd.DataFrame, col_name: str, label_name: str = LABEL,
                                                  is_train_name: str = 'SRC') -> pd.DataFrame:
    kf = KFold(n_splits=5, shuffle=True, random_state=active_random_state)
    tmp_data = data[[col_name, label_name, is_train_name]].copy()
    result = tmp_data[[col_name, is_train_name]].copy()
    result[col_name] = 0

    target_encoder = TargetEncoder(cols=[col_name], handle_unknown='value', handle_missing='value')

    for train_idx, valid_idx in kf.split(tmp_data.loc[(tmp_data[is_train_name] == 'train')]):
        X_train, X_valid = tmp_data.iloc[train_idx], tmp_data.iloc[valid_idx]
        y_train = tmp_data[label_name].iloc[train_idx]

        target_encoder.fit(X_train[col_name], y_train)
        result.loc[valid_idx, col_name] = target_encoder.transform(X_valid[col_name]).loc[:, col_name]
        result.loc[(result[is_train_name] == 'test'), col_name] = (
                    result.loc[(result[is_train_name] == 'test'), col_name] +
                    target_encoder.transform(tmp_data.loc[(tmp_data[is_train_name] == 'test'), col_name]).loc[:, col_name] / 5)

    return result[col_name]


def category_encoding_by_catboost(data: pd.DataFrame, col_name: str, label_name: str = LABEL,
                                  is_train_name: str = 'SRC') -> pd.DataFrame:
    """
    类别变量catboost encoding处理
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param label_name: 标签类名
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, label_name, is_train_name]].copy()
    encoder = CatBoostEncoder(cols=[col_name], handle_unknown='value', handle_missing='value') \
        .fit(tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name],
             tmp_data.loc[(tmp_data[is_train_name] == 'train'), label_name])  # 在训练集上训练
    return encoder.transform(tmp_data[col_name])


def binary_cross_columns(data: pd.DataFrame, cross_cols: List[List]) -> pd.DataFrame:
    """
    类别变量交叉组合
    :param data: 原始数据集
    :param cross_cols: like [['fea1', 'fea2'], ['fea3', 'fea4']], 类型要求是objects(str)
    :return: 生成的新的特征集
    """
    all_cols = set([cols for cross_item in cross_cols for cols in cross_item])
    tmp_data = data[all_cols].copy()

    col_names = ['_'.join(cross_item) for cross_item in cross_cols]
    cross_dict = {k: v for k, v in zip(col_names, cross_cols)}

    for k, v in cross_dict.items():
        tmp_data[k] = tmp_data[v].apply(lambda x: '-'.join(x), axis=1)

    return tmp_data[col_names]


def lower_quartile(x):
    """
    下四分位数
    """
    return x.quantile(0.25)


def upper_quartile(x):
    """
    上四分位数
    """
    return x.quantile(0.75)


def group_by_statistic(data: pd.DataFrame, key_cols: list,
                       num_aggregations: dict, cat_aggregations: dict, prefix: str) -> pd.DataFrame:
    """
    group by 统计函数
    :param data: 原始数据集
    :param key_cols: group by keys
    :param num_aggregations: 连续变量, eg.{'fea1': ['min', 'max', 'mean', 'var', 'median', 'skew', 'count', lower_quartile, upper_quartile]}
    :param cat_aggregations: 离散变量, eg.{'fea2': ['min', 'max', 'mean', 'var', 'median', 'count']}
    :param prefix: 新特征列名的prefix，如表名
    :return: 生成的新的特征集，包括key columns
    """
    cols_all = key_cols + list(num_aggregations.keys()) + list(cat_aggregations.keys())
    tmp_data = data[cols_all].copy()
    group_by_result = tmp_data.groupby(by=key_cols).agg({**num_aggregations, **cat_aggregations}).reset_index()
    group_by_cols_name = key_cols
    for k, v in {**num_aggregations, **cat_aggregations}.items():
        for agg in v:
            if isfunction(agg):
                agg = agg.__name__
            group_by_cols_name.append(prefix + '_' + k + '_' + agg)
    group_by_result.columns = group_by_cols_name
    return group_by_result


ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def derived_features_between_cols(data: pd.DataFrame, cols_and_ops: List[list]) -> pd.DataFrame:
    """
    两列间的运算
    :param data: 原始数据集
    :param cols_and_ops: eg. [['fea1', 'fea2', '-'], ['fea3', 'fea4', '/']], 操作符字典ops
    :return: 生成的新的特征集
    """
    cols_all = []
    for col_and_op in cols_and_ops:
        cols_all.extend(col_and_op[0:2])
    tmp_data = data[cols_all].copy()

    features_new = pd.DataFrame()
    for col_and_op in cols_and_ops:
        col_name_new = col_and_op[0] + col_and_op[2] + col_and_op[1]
        features_new[col_name_new] = ops[col_and_op[2]](tmp_data[col_and_op[0]], tmp_data[col_and_op[1]])
    return features_new


def calculate_age(age):
    if age <= 20:
        return 20
    if 20 < age <= 30:
        return 25
    if 30 < age <= 40:
        return 35
    if 40 < age <= 50:
        return 45
    if 50 < age <= 60:
        return 55
    if 60 < age <= 70:
        return 65
    if 70 < age <= 80:
        return 75
    if 80 < age <= 90:
        return 85
    if age > 90:
        return 95

#
# if __name__ == '__main__':
#     df_data = loader.to_df(Path(dir_preprocess).joinpath(f'flatmap.csv'))
#     df_target = loader.to_df_label()
#     df_data = df_data.merge(df_target, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left')
#     res = category_encoding_by_cross_val_target_encoder(df_data, 'NTRL_RANK_CD')
#     print(res)
