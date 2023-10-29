from typing import List
import os
import pickle

from pathlib import Path
import numpy as np
from sklearn.utils import Bunch
import pandas as pd

# data_path = 'D:/workspace/document/first_loan/processing/first'
# train_file_name = 'train2.csv'
# predict_file_name = "predict2.csv"
# train_file_path = os.path.join(data_path, train_file_name)
# predict_file_path = os.path.join(data_path, predict_file_name)
#
# result_data_path = PATH = Path('../data/')
# result_train_file_name = 'train.p'
# result_predict_file_name = "predict.p"

target_col = 'LABEL'
id_col = 'ID_UNI'


data_path = 'D:/workspace/work/first_loan/data'
train_file_name = 'penguins_processed.csv'
predict_file_name = "penguins_processed.csv"
train_file_path = os.path.join(data_path, train_file_name)
predict_file_path = os.path.join(data_path, predict_file_name)

result_data_path = PATH = Path('../data/')
result_train_file_name = 'train_multi.p'
result_predict_file_name = "predict_multi.p"


def create_data_bunch(category_cols: List = None) -> None:
    if category_cols is None:
        category_cols = []
    data_bunch_train = Bunch()
    train_data = pd.read_csv(train_file_path, index_col=None)

    data_bunch_train.target = train_data[target_col]
    train_data.drop(target_col, axis=1, inplace=True)
    data_bunch_train.data = train_data
    data_bunch_train.col_names = train_data.columns.tolist()
    data_bunch_train.category_cols = category_cols
    pickle.dump(data_bunch_train, open(PATH / result_train_file_name, "wb"))

    # data_bunch_predict = Bunch()
    # predict_data = pd.read_csv(predict_file_path, index_col=None)
    # data_bunch_predict.id = predict_data[id_col]
    # predict_data.drop(id_col, axis=1, inplace=True)
    # data_bunch_predict.data = predict_data
    # pickle.dump(data_bunch_predict, open(PATH / result_predict_file_name, "wb"))

    data_bunch_predict = Bunch()
    predict_data = pd.read_csv(predict_file_path, index_col=None)
    data_bunch_predict.id = pd.Series(np.arange(predict_data.shape[0]))
    predict_data.drop(target_col, axis=1, inplace=True)
    data_bunch_predict.data = predict_data
    pickle.dump(data_bunch_predict, open(PATH / result_predict_file_name, "wb"))


# create_data_bunch()
create_data_bunch(['island', 'sex'])
