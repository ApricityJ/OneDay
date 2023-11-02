from typing import Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from constant import active_random_state
from data import loader


def adv_val(X_adv: DataFrame, y_adv: DataFrame):

    accs = []
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
        accs.append(acc)

    # 输出平均准确率
    avg_acc = np.mean(accs)
    print(f'average accuracy: {avg_acc:.2f}')
    lgb.plot_importance(model)
    plt.show()
    plt.savefig()


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


def do_adv_val():
    # 加载数据集
    # ds: Dict[str, list[DataFrame]] = loader.to_ds_train_test()
    #
    # for k, v in ds.items():
    #     print(f'process -> {k}')
    #
    #     X_train, X_test = drop_distractions(v[0], v[1])
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

    v = loader.to_df_train_test('XW_AGET_PAY')
    X_train, X_test = drop_distractions(v[0],v[1])

    print(len(X_train))
    print(len(X_test))

    # 创建标签
    y_adv_train = [0] * len(X_train)
    y_adv_test = [1] * len(X_test)

    # 特征和标签
    X_adv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
    y_adv = pd.concat([pd.Series(y_adv_train), pd.Series(y_adv_test)], axis=0)

    # 对抗验证
    adv_val(X_adv, y_adv)


if __name__ == '__main__':
    do_adv_val()
