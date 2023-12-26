from typing import Dict
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from constant import *
from data import loader

pd.set_option('display.max_rows', None)



def adv_val(X_adv: DataFrame, y_adv: DataFrame):

    accs = []
    f1s = []
    aucs = []
    model = lgb.LGBMClassifier()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=active_random_state)
    for train_index, test_index in skf.split(X_adv, y_adv):

        X_adv_train, X_adv_test = X_adv.iloc[train_index], X_adv.iloc[test_index]
        y_adv_train, y_adv_test = y_adv.iloc[train_index], y_adv.iloc[test_index]

        # 训练模型
        model.fit(X_adv_train, y_adv_train)

        # 预测
        y_adv_pred = model.predict(X_adv_test)

        # 计算准确率
        acc = accuracy_score(y_adv_test, y_adv_pred)
        f1 = f1_score(y_adv_test, y_adv_pred)
        auc = roc_auc_score(y_adv_test, y_adv_pred)
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

    # 输出平均准确率
    print(accs)
    print(f1s)
    print(aucs)
    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)
    avg_auc = np.mean(aucs)
    print(f'average accuracy: {avg_acc:.2f}')
    print(f'average f1: {avg_f1:.2f}')
    print(f'average auc: {avg_auc:.2f}')
    # lgb.plot_importance(model)
    # plt.show()
    # plt.savefig()
    feature_importance = pd.DataFrame({
        'column': X_adv.columns,
        'importance': model.feature_importances_,
    }).sort_values(by='importance', ascending=False)
    print(feature_importance)
    return feature_importance



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


data_bunch = pickle.load(open(Path(dir_train).joinpath(file_name_train), 'rb'))
X_train = data_bunch.data
data_bunch = pickle.load(open(Path(dir_test).joinpath(file_name_test), 'rb'))
X_test = data_bunch.data

# X_train = X_train.sample(X_test.shape[0])

print(X_train.shape)
print(X_test.shape)

# columns = list(X_train.columns)
# selected = []
# for item in columns:
#     if '_minus_' in item or '_div_' in item:
#         continue
#     selected.append(item)
#
# print(selected)
# X_train = X_train[selected]
# X_test = X_test[selected]
#
#
# print(X_train.shape)
# print(X_test.shape)


# 创建标签
y_adv_train = [0] * len(X_train)
y_adv_test = [1] * len(X_test)

# 特征和标签
X_adv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
y_adv = pd.concat([pd.Series(y_adv_train), pd.Series(y_adv_test)], axis=0)

# 对抗验证
feature_importance = adv_val(X_adv, y_adv)


feature_importance_drop = feature_importance[feature_importance['importance'] >= 10]
feature_importance_drop = feature_importance_drop['column']

X_train.drop(columns=feature_importance_drop, inplace=True)
X_test.drop(columns=feature_importance_drop, inplace=True)

X_adv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
y_adv = pd.concat([pd.Series(y_adv_train), pd.Series(y_adv_test)], axis=0)
feature_importance = adv_val(X_adv, y_adv)





# def do_adv_val():
#     # 加载数据集
#     # ds: Dict[str, list[DataFrame]] = loader.to_ds_train_test()
#     #
#     # for k, v in ds.items():
#     #     print(f'process -> {k}')
#     #
#     #     X_train, X_test = drop_distractions(v[0], v[1])
#     #
#     #     # 创建标签
#     #     y_adv_train = [0] * len(X_train)
#     #     y_adv_test = [1] * len(X_test)
#     #
#     #     # 特征和标签
#     #     X_adv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
#     #     y_adv = pd.concat([pd.Series(y_adv_train), pd.Series(y_adv_test)], axis=0)
#     #
#     #     # 对抗验证
#     #     adv_val(X_adv, y_adv)
#
#     # v = loader.to_df_train_test('XW_AGET_PAY')
#     # X_train, X_test = drop_distractions(v[0],v[1])
#     data_bunch = pickle.load(open(Path(dir_train).joinpath(file_name_train), 'rb'))
#     X_train = data_bunch.data
#     data_bunch = pickle.load(open(Path(dir_test).joinpath(file_name_test), 'rb'))
#     X_test = data_bunch.data
#
#     # print(len(X_train))
#     # print(len(X_test))
#     print(X_train.shape)
#     print(X_test.shape)
#
#     # 创建标签
#     y_adv_train = [0] * len(X_train)
#     y_adv_test = [1] * len(X_test)
#
#     # 特征和标签
#     X_adv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
#     y_adv = pd.concat([pd.Series(y_adv_train), pd.Series(y_adv_test)], axis=0)
#
#     # 对抗验证
#     adv_val(X_adv, y_adv)
