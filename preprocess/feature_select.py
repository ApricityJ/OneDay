from pathlib import Path

from boruta import BorutaPy
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data import loader, exporter
from util import jsons
from constant import *

import warnings

warnings.filterwarnings('ignore')


def load_dataframe_to_process(key: str):
    df_data = loader.to_df(Path(dir_preprocess).joinpath(f'{key}.csv'))
    return df_data[df_data['SRC'] == 'train']


def select_by_boruta(key: str):
    df_data = load_dataframe_to_process(key)

    df_target = loader.to_df_label()

    df_data = df_data.merge(df_target, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left')
    label_col = df_data[LABEL]
    df_data.drop([LABEL, 'SRC', 'CUST_NO'], axis=1, inplace=True)
    print(f"column nums : {df_data.shape[1]}")

    estimator = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    # estimator = LGBMClassifier(n_estimators=100, n_jobs=-1, verbose=0, num_boost_round=100)  # 有问题的
    # estimator = LGBMClassifier(n_jobs=-1, max_depth=5, num_leaves=31)

    # 寻找所有相关的特征
    boruta = BorutaPy(estimator=estimator, n_estimators="auto", verbose=2, random_state=active_random_state)

    boruta.fit(df_data.values, label_col.values)
    selected = df_data.columns[boruta.support_]
    print(f'select column nums : {len(selected)}')
    selected = selected.tolist()
    selected.insert(0, 'CUST_NO')
    jsons.to_json(list(selected), Path(dir_result).joinpath(f'{key}_selected_cols.json'))


def select_by_boruta_result(key: str):
    pass




def unique_values_check(df):
    """
    检查 DataFrame 中的每一列，如果某列只包含唯一值，则将该列的名称加入列表中。

    :param df: 要检查的 DataFrame
    :return: 只包含唯一值的列的名称列表
    """
    columns_with_unique_values = []

    # 遍历每一列并检查是否只包含唯一值
    for column in df.columns:
        if df[column].nunique() == 1:
            columns_with_unique_values.append(column)

    print(f'unique check - columns to delete = {len(columns_with_unique_values)}')
    print(columns_with_unique_values)
    print('-----------------------------')

    return columns_with_unique_values


def high_null_percentage_check(df, threshold):
    """
    检查 DataFrame 中的每一列，如果某列的空值数占总数的百分比超过了阈值，则将该列的名称加入列表中。

    :param df: 要检查的 DataFrame
    :param threshold: 空值百分比的阈值
    :return: 超过阈值的列的名称列表
    """
    columns_with_high_null = []

    # 遍历每一列并计算空值百分比
    for column in df.columns:
        null_percentage = df[column].isnull().sum() / len(df)
        if null_percentage > threshold:
            columns_with_high_null.append(column)

    print(f'null check - threshold = {threshold} - columns to delete = {len(columns_with_high_null)}')
    print(columns_with_high_null)
    print('-----------------------------')

    return columns_with_high_null


def correlation_check(df, threshold):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            print(f'{column}: {upper[upper[column] > threshold].index.values}')
            to_drop.append(column)

    print(f'correlation check - threshold = {threshold}, columns to delete = {len(to_drop)}')
    print(to_drop)
    print('-----------------------------')
    return to_drop


def base_select(key: str):
    df_data = load_dataframe_to_process(key)
    df_data.drop(['SRC', 'CUST_NO'], axis=1, inplace=True)
    print(f"column nums : {df_data.shape[1]}")

    columns_to_drop = []
    columns_with_unique_values = unique_values_check(df_data)
    columns_to_drop.extend(columns_with_unique_values)
    columns_with_high_null = high_null_percentage_check(df_data, 0.8)
    columns_to_drop.extend(columns_with_high_null)
    columns_with_high_correlation = correlation_check(df_data, 0.95)
    columns_to_drop.extend(columns_with_high_correlation)

    print(f"after column nums : {df_data.shape[1] - len(columns_to_drop)}")
    jsons.to_json(columns_to_drop, Path(dir_result).joinpath(f'{key}_base_to_drop.json'))
    return columns_to_drop




columns_to_drop = base_select('flatmap')


# def plot_correlation_matrix(df):
#     # 计算相关性矩阵
#     corr_matrix = df.corr()
#
#     # 设置 matplotlib 图形大小
#     plt.figure(figsize=(10, 8))
#
#     # 使用 seaborn 绘制热图
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
#
#     # 显示图形
#     plt.show()


# def calculate_iv(df, feature, target):
#     """
#     计算特征相对于目标变量的信息价值 (IV)。
#     """
#     # 创建交叉表
#     contingency_table = pd.crosstab(df[feature], df[target])
#
#     # 计算每个分组的好坏比例
#     total_bad = contingency_table[1].sum()
#     total_good = contingency_table[0].sum()
#     contingency_table['Good%'] = contingency_table[0] / total_good
#     contingency_table['Bad%'] = contingency_table[1] / total_bad
#
#     # 避免除以零
#     contingency_table['Good%'] = np.where(contingency_table['Good%'] == 0, np.nan, contingency_table['Good%'])
#     contingency_table['Bad%'] = np.where(contingency_table['Bad%'] == 0, np.nan, contingency_table['Bad%'])
#
#     # 计算 WOE 和 IV
#     contingency_table['WOE'] = np.log(contingency_table['Good%'] / contingency_table['Bad%'])
#     contingency_table['IV'] = (contingency_table['Good%'] - contingency_table['Bad%']) * contingency_table['WOE']
#
#     # 返回 IV 总和
#     return contingency_table['IV'].sum()


# def remove_features_based_on_iv_and_correlation(df, target, threshold=0.9):
#     # 计算相关性矩阵
#     corr_matrix = df.corr().abs()
#
#     # 找出高度相关的特征对
#     high_corr_pairs = [(i, j) for i in range(len(corr_matrix.columns)) for j in range(i+1, len(corr_matrix.columns))
#                        if corr_matrix.iloc[i, j] > threshold and corr_matrix.columns[i] != target and corr_matrix.columns[j] != target]
#
#     # 选择要删除的特征
#     features_to_remove = set()
#     for i, j in high_corr_pairs:
#         feature_i = corr_matrix.columns[i]
#         feature_j = corr_matrix.columns[j]
#
#         # 计算两个特征的 IV 值
#         iv_i = calculate_iv(df, feature_i, target)
#         iv_j = calculate_iv(df, feature_j, target)
#
#         # 删除 IV 值较低的特征
#         if iv_i < iv_j:
#             features_to_remove.add(feature_i)
#         else:
#             features_to_remove.add(feature_j)
#
#     # 删除选定的特征
#     # df_reduced = df.drop(columns=list(features_to_remove))
#     print(features_to_remove)
#     print(len(features_to_remove))
#
#     # return df_reduced