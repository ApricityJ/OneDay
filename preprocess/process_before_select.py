import pandas as pd
from category_encoders import OneHotEncoder

from utils.loader import *
from utils.exporter import *
from constant import *


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


def category_encoding_by_onehot(data: pd.DataFrame, col_name: str, is_train_name: str = 'IS_TRAIN') -> pd.DataFrame:
    """
    类别变量onehot化
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()
    encoder = OneHotEncoder(cols=[col_name], use_cat_names=True)
    encoder.fit(tmp_data.loc[(tmp_data[is_train_name] == 'train'), col_name])
    return encoder.transform(tmp_data[col_name])


# 处理 preprocess文件夹中的flatmap.csv，并保存到train、test目录的 train_flatmap.csv、test_flatmap.csv
def preprocess_flatmap(key='flatmap'):
    df_all = to_df(Path(dir_preprocess).joinpath(f'{key}.csv'))
    # df_all = df_all.iloc[:, 0:10]

    df_all['NTRL_CUST_AGE'] = df_all['NTRL_CUST_AGE'].apply(calculate_age)
    df_all['NTRL_CUST_SEX_CD'] = df_all['NTRL_CUST_SEX_CD'].apply(lambda x: 0 if x == 'A' else 1)
    result_rank_cd = category_encoding_by_onehot(df_all, 'NTRL_RANK_CD', is_train_name = 'SRC')
    df_all = pd.concat([df_all, result_rank_cd], axis=1)
    df_all.drop(['NTRL_RANK_CD'], axis=1, inplace=True)
    df_all.drop(['DATA_DAT'], axis=1, inplace=True)
    df_all = df_all.fillna(0)

    df_test = df_all[df_all['SRC'] == 'test']
    df_train = df_all[df_all['SRC'] == 'train']
    df_train.drop(['SRC'], axis=1, inplace=True)
    df_test.drop(['SRC'], axis=1, inplace=True)

    df_label = to_df(Path(dir_train).joinpath(f'TARGET_QZ.csv'))
    df_label.drop(['DATA_DAT', 'CARD_NO'], axis=1, inplace=True)
    df_train = pd.merge(df_train, df_label, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left')
    df_train.drop(['CUST_NO'], axis=1, inplace=True)
    print(df_train.columns)
    # print(df_test.columns)
    # print(df_train['FLAG'].isnull().sum())

    export_df_to_train(f'train_{key}', df_train)
    export_df_to_test(f'test_{key}', df_test)


def preprocess_all():
    preprocess_flatmap('flatmap')

preprocess_all()
print("--------- done preprocess data ---------")