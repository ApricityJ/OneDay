from pathlib import Path
import pickle

from boruta import BorutaPy
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from data import loader, exporter
from util import jsons
from constant import *
import util.metrics as metrics

import warnings

warnings.filterwarnings('ignore')


# 读取数据
# data_bunch = pickle.load(open(Path(dir_train).joinpath(file_name_train), 'rb'))
# X_train = data_bunch.data
# data_bunch = pickle.load(open(Path(dir_test).joinpath(file_name_test), 'rb'))
# X_test = data_bunch.data


#
# from boruta import BorutaPy
# from sklearn.ensemble import RandomForestClassifier
# from constant import active_random_state
# from data import loader
#
#
# def select_by_boruta():
#     train, _ = loader.to_df_train_test()
#
#     rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
#     boruta = BorutaPy(estimator=rf, n_estimators="auto", verbose=2, random_state=active_random_state)
#
#     boruta.fit(train[:-1], train['label'])
#
#     return boruta.transform(train)
#
#
# def select_by_wrapper():
#     pass


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


def distribution_check(df_train, df_test, columns, threshold):
    to_drop = []
    for col in columns:
        result = stats.ks_2samp(df_train[col], df_test[col])
        if result.pvalue < threshold:
            to_drop.append(col)

    print(f'distribution check - threshold = {threshold}, columns to delete = {len(to_drop)}')
    print(to_drop)
    print('-----------------------------')
    return to_drop


# 这里load的文件是包含 'SRC' 'ID', 但没有 'LABEL' 的
def load_dataframe_to_process(key: str):
    df_data = loader.to_df(Path(dir_preprocess).joinpath(f'{key}.csv'))
    # print(df_data.columns.tolist())
    if LABEL in df_data.columns.tolist():
        df_data.drop(columns=[LABEL], inplace=True)
    return df_data[df_data['SRC'] == 'train'], df_data[df_data['SRC'] == 'test']


# 删除扰动项，主要有日期和object类型
def drop_distractions(df_train: DataFrame, df_test: DataFrame):
    idx_to_drop = []
    for idx, type in df_train.dtypes.items():
        if type == 'object':
            idx_to_drop.append(idx)

    print(f'drop columns -> {idx_to_drop}')
    df_train.drop(idx_to_drop, axis=1, inplace=True)
    df_test.drop(idx_to_drop, axis=1, inplace=True)

    return df_train, df_test


def adv_val(X_adv: DataFrame, y_adv: DataFrame):
    # accs = []
    # f1s = []
    aucs = []
    model = lgb.LGBMClassifier()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=active_random_state)
    for train_index, test_index in skf.split(X_adv, y_adv):
        X_adv_train, X_adv_test = X_adv.iloc[train_index], X_adv.iloc[test_index]
        y_adv_train, y_adv_test = y_adv.iloc[train_index], y_adv.iloc[test_index]

        # 训练模型
        model.fit(X_adv_train, y_adv_train)

        # 预测
        y_adv_pred = model.predict_proba(X_adv_test)[:,1]

        # acc = accuracy_score(y_adv_test, y_adv_pred)
        # f1 = f1_score(y_adv_test, y_adv_pred)
        auc = roc_auc_score(y_adv_test, y_adv_pred)
        # accs.append(acc)
        # f1s.append(f1)
        aucs.append(auc)

    # 输出平均准确率
    # print(accs)
    # print(f1s)
    print(aucs)
    # avg_acc = np.mean(accs)
    # avg_f1 = np.mean(f1s)
    avg_auc = np.mean(aucs)
    print(f'average auc: {avg_auc:.2f}')  # 小于0.7
    # lgb.plot_importance(model)
    # plt.show()
    # plt.savefig()
    feature_importance = pd.DataFrame({
        'column': X_adv.columns,
        'importance': model.feature_importances_,
    }).sort_values(by='importance', ascending=False)
    print(feature_importance[:50])
    return feature_importance


def adv_val_select(key: str):
    df_train, df_test = load_dataframe_to_process(key)
    drop_cols = jsons.of_json(Path(dir_result).joinpath(f'{key}_base_to_drop.json'))
    df_train.drop(['SRC', ID, ], axis=1, inplace=True)  # 如果里面有日期也要记得删除
    df_test.drop(['SRC', ID, ], axis=1, inplace=True)
    df_train.drop(drop_cols, axis=1, inplace=True)
    df_test.drop(drop_cols, axis=1, inplace=True)
    print(f"train column nums : {df_train.shape}")
    print(f"test column nums : {df_test.shape}")
    print(df_train.columns.tolist())

    df_train, df_test = drop_distractions(df_train, df_test)

    # 创建标签
    y_adv_train = [0] * len(df_train)
    y_adv_test = [1] * len(df_test)

    # 特征和标签
    X_adv = pd.concat([pd.DataFrame(df_train), pd.DataFrame(df_test)], axis=0)
    y_adv = pd.concat([pd.Series(y_adv_train), pd.Series(y_adv_test)], axis=0)

    # 对抗验证
    feature_importance = adv_val(X_adv, y_adv)

    # feature_importance_drop = feature_importance[feature_importance['importance'] >= 10]
    # feature_importance_drop = feature_importance_drop['column']
    #
    # X_train.drop(columns=feature_importance_drop, inplace=True)
    # X_test.drop(columns=feature_importance_drop, inplace=True)
    #
    # X_adv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
    # y_adv = pd.concat([pd.Series(y_adv_train), pd.Series(y_adv_test)], axis=0)
    # feature_importance = adv_val(X_adv, y_adv)


def base_select(key: str):
    df_train, df_test = load_dataframe_to_process(key)
    df_train.drop(['SRC', ID], axis=1, inplace=True)
    df_test.drop(['SRC', ID], axis=1, inplace=True)
    print(f"train column nums : {df_train.shape[1]}")
    print(f"test column nums : {df_test.shape[1]}")

    columns_to_drop = []
    columns_with_unique_values = unique_values_check(df_train)
    columns_to_drop.extend(columns_with_unique_values)
    # columns_with_high_null = high_null_percentage_check(df_train, 0.97)
    # columns_to_drop.extend(columns_with_high_null)
    # columns_with_high_correlation = correlation_check(df_train, 0.97)
    # columns_to_drop.extend(columns_with_high_correlation)
    columns_to_check_distribution = list(set(df_train.columns.tolist()) - set(columns_to_drop))
    columns_with_diff_distribution = distribution_check(df_train, df_test, columns_to_check_distribution, 0.05)
    columns_to_drop.extend(columns_with_diff_distribution)

    print(f"after column nums : {df_train.shape[1] - len(columns_to_drop)}")
    return columns_to_drop


def boruta_select(key: str):
    # 读取数据和LABEL
    df_train, df_test = load_dataframe_to_process(key)
    df_target = loader.to_df_label()

    # 拼接
    df_data = df_train.merge(df_target, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left')
    label_col = df_data[LABEL]
    df_data.drop([LABEL, 'SRC', 'CUST_NO'], axis=1, inplace=True)
    # 要先处理好类别特征
    df_data.drop(['NTRL_CUST_SEX_CD', 'NTRL_RANK_CD'], axis=1, inplace=True)
    # df_data = df_data.iloc[:, :50]  # 仅用于测试
    print(f"column nums : {df_data.shape[1]}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df_data, label_col, test_size=0.2,
                                                        random_state=active_random_state)

    X_train.fillna(-999, inplace=True)
    # 使用LightGBM的分类模型作为Boruta的基础模型
    # estimator = LGBMClassifier(n_estimators=100, num_boost_round=100, random_state=42, n_jobs=-1)
    estimator = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    # estimator = LGBMClassifier(n_estimators=100, n_jobs=-1, verbose=0, num_boost_round=100)  # 有问题的
    # estimator = LGBMClassifier(n_jobs=-1, max_depth=5, num_leaves=31)

    # 寻找所有相关的特征
    # boruta = BorutaPy(estimator=estimator, n_estimators="auto", verbose=2, random_state=active_random_state)

    # 创建Boruta特征选择器
    boruta_selector = BorutaPy(
        estimator,
        n_estimators='auto',  # 自动确定最优树数量
        verbose=2,
        random_state=active_random_state
    )

    # 特征选择
    boruta_selector.fit(X_train.values, y_train.values)

    # 被Boruta选中的特征
    selected_features = X_train.columns[boruta_selector.support_].to_list()

    print("Selected Features nums :", len(selected_features))
    print("Selected Features by Boruta:", selected_features)

    # 保存为文件
    selected = selected_features
    selected.insert(0, 'CUST_NO')
    jsons.to_json(list(selected), Path(dir_result).joinpath(f'{key}_selected_cols_by_boruta.json'))


def lgb_select(key: str, num_runs: int, threshold: float):
    # 读取数据和LABEL
    df_train, df_test = load_dataframe_to_process(key)
    drop_cols = jsons.of_json(Path(dir_result).joinpath(f'{key}_base_to_drop.json'))
    df_train.drop(drop_cols, axis=1, inplace=True)
    df_test.drop(drop_cols, axis=1, inplace=True)
    df_target = loader.to_df_label()

    # 拼接
    X = df_train.merge(df_target, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left')
    y = X[LABEL]
    X.drop([LABEL, 'SRC', 'CUST_NO'], axis=1, inplace=True)
    print(f"column nums : {X.shape[1]}")

    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns

    for random_state in random_states[:num_runs]:
        # 打乱特征顺序
        shuffled_features = np.random.permutation(X.columns)
        X_shuffled = X[shuffled_features]

        # 分割数据集
        X_train, X_valid, y_train, y_valid = train_test_split(X_shuffled, y, test_size=0.2,
                                                              random_state=active_random_state)

        # 定义LightGBM模型
        # model = LGBMClassifier(random_state=random_state, n_estimators=1000, learning_rate=0.1)
        # model.fit(
        #     X_train, y_train,
        #     eval_set=[(X_valid, y_valid)],
        #     eval_metric=getattr(metrics, 'lgb_ks_score_eval'),
        #     early_stopping_rounds=400,
        # )

        params = {'random_state': random_state, 'n_estimators': 1000, 'learning_rate': 0.01}
        model = lgb.train(params,
                          lgb.Dataset(X_train, y_train),
                          feval=getattr(metrics, 'lgb_ks_score_eval'),
                          valid_sets=[lgb.Dataset(X_valid, y_valid)],
                          early_stopping_rounds=400,
                          verbose_eval=1)

        # 获取特征重要性并保存
        fold_importance = model.feature_importance()
        feature_importances[f'run_{random_state}'] = fold_importance
        # print(feature_importances)

    # 平均一下
    print(feature_importances)
    feature_importances['mean_importance'] = feature_importances.iloc[:, 1:].mean(axis=1)

    selected_features = feature_importances[feature_importances['mean_importance'] > threshold]['feature'].tolist()

    print("Selected Features nums :", len(selected_features))
    print("Selected Features by lgb:", selected_features)

    # 保存为文件
    selected = selected_features
    selected.insert(0, 'CUST_NO')
    jsons.to_json(list(selected), Path(dir_result).joinpath(f'{key}_selected_cols_by_lgb.json'))


# 基础特征筛选
key_1 = 'v5'
columns_to_drop_base = base_select(key_1)
jsons.to_json(columns_to_drop_base, Path(dir_result).joinpath(f'{key_1}_base_to_drop.json'))
print('-----------------------------')

# 对抗验证
# adv_val_select(key_1)
print('-----------------------------')

# lgb_select('v6', 3, 0)
# boruta_select('flatmap')
