from typing import List
import os
import pickle

from pathlib import Path
import numpy as np
from sklearn.utils import Bunch
import pandas as pd

from data import loader
from util import jsons
from constant import *


# 拿flatmap做个测试
# flatmap_selected_cols = jsons.of_json(Path(dir_result).joinpath(f'flatmap_selected_cols.json'))
# flatmap_selected_cols.append("SRC")
# df_data = loader.to_df(Path(dir_preprocess).joinpath(f'flatmap.csv'))
# df_data = df_data[flatmap_selected_cols]

drop_cols = jsons.of_json(Path(dir_result).joinpath(f'train_base_to_drop.json'))
df_train = loader.to_df(Path(dir_preprocess).joinpath(f'train.csv'))
df_test = loader.to_df(Path(dir_preprocess).joinpath(f'test.csv'))
df_train = df_train.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)
print(df_train.shape)
print(df_test.shape)

# df_target = loader.to_df_label()
# df_data = df_data.merge(df_target, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left')

# train = df_data[df_data['SRC'] == 'train']
# train = train.drop(columns=['SRC'])

# test = df_data[df_data['SRC'] == 'test']
# test = test.drop(columns=['SRC'])


category_cols = []
data_bunch_train = Bunch()
data_bunch_train.target = df_train[LABEL]
df_train.drop([LABEL, ID], axis=1, inplace=True)
data_bunch_train.data = df_train
data_bunch_train.col_names = df_train.columns.tolist()
data_bunch_train.category_cols = category_cols
pickle.dump(data_bunch_train, open(Path(dir_train).joinpath("train.p"), "wb"))

data_bunch_predict = Bunch()
data_bunch_predict.id = df_test[ID]
df_test.drop([ID], axis=1, inplace=True)
data_bunch_predict.data = df_test
pickle.dump(data_bunch_predict, open(Path(dir_test).joinpath("test.p"), "wb"))
