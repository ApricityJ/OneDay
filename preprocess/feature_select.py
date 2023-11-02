from typing import List
from pathlib import Path

from boruta import BorutaPy
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch
import pickle

from preprocessing.process_before_select import preprocess_all
from utils import loader
from constant import *


def select_by_boruta(key: str):
    # preprocess_all(key)
    # train, test = loader.to_df_train_test()
    train = loader.to_df(Path(dir_train).joinpath(f'train_{key}.csv'))
    test = loader.to_df(Path(dir_test).joinpath(f'test_{key}.csv'))

    label_col = train[LABEL]
    train.drop(LABEL, axis=1, inplace=True)
    test_id_col = test[ID]

    estimator = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    # estimator = LGBMClassifier(n_estimators=1000, n_jobs=-1, verbose=0)
    boruta = BorutaPy(estimator=estimator, n_estimators="auto", verbose=2, random_state=active_random_state)

    boruta.fit(train.values, label_col.values)
    selected = train.columns[boruta.support_]
    print(f'select columns : {selected}')

    return  train[selected], label_col, test[selected], test_id_col


def create_data_bunch(train_data, train_label, test_data, test_id, category_cols: List = None) -> None:
    if category_cols is None:
        category_cols = []
    data_bunch_train = Bunch()

    data_bunch_train.target = train_label
    data_bunch_train.data = train_data
    data_bunch_train.col_names = train_data.columns.tolist()
    data_bunch_train.category_cols = category_cols
    pickle.dump(data_bunch_train, open(Path(dir_train).joinpath(file_name_train), "wb"))

    data_bunch_predict = Bunch()
    data_bunch_predict.id = test_id
    data_bunch_predict.data = test_data
    pickle.dump(data_bunch_predict, open(Path(dir_test).joinpath(file_name_test), "wb"))


def select_feature_and_prepare_data(key: str):
    train_data, train_label, test_data, test_id = select_by_boruta(key)
    print("--------- done feature select ---------")
    category_cols = [] # 可以指定哪几列是类别变量
    create_data_bunch(train_data, train_label, test_data, test_id, category_cols)
    print("--------- done prepare train and test data ---------")
