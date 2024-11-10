import warnings
from pathlib import Path
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import Bunch

from constant import *
from util.metrics import lgb_ks_score_eval
from data import exporter

warnings.filterwarnings('ignore', category=UserWarning)

random_state = 42

# 超参数空间
param_space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.05),  # 学习率
    'num_leaves': hp.choice('num_leaves', np.arange(20, 200, dtype=int)),  # 叶子数
    'max_depth': hp.choice('max_depth', np.arange(3, 12, dtype=int)),  # 树的最大深度
    'min_child_weight': hp.uniform('min_child_weight', 0.001, 10),  # 子叶节点的最小权重
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),  # 样本列采样率
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 20),
    'subsample': hp.uniform('subsample', 0.5, 1.0),  # 样本采样率
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),  # L1正则化
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)  # L2正则化
}


def ks_stat(y_true, y_pred):
    """计算KS值的自定义评估函数"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)  # 计算ROC曲线
    ks_value = np.max(np.abs(tpr - fpr))  # KS统计量
    return ks_value


# 自定义目标函数，用于贝叶斯优化
def objective(params, X, y, n_folds=5):
    """贝叶斯优化的目标函数，返回负的整体验证集KS分数"""
    # 设置模型参数
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['verbose'] = -1
    params['n_estimators'] = 10000
    params['metric'] = 'auc'
    params['num_leaves'] = params['num_leaves']
    params['max_depth'] = params['max_depth']
    params['seed'] = random_state
    params['bagging_seed'] = random_state
    params['feature_fraction_seed'] = random_state
    params['data_random_seed'] = random_state
    params['deterministic'] = True
    params['force_col_wise'] = True  # Helps in achieving determinism
    # params['num_threads'] = 1        # Single thread for reproducibility

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # 存储每一折的预测结果
    final_predictions = np.zeros(len(X))

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # 建立LightGBM训练集
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        # 训练LightGBM模型
        model = lgb.train(
            params,
            lgb_train,
            feval=lgb_ks_score_eval,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=400),
                lgb.log_evaluation(100)
            ]
        )

        # 验证集预测概率
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

        # 将每一折的预测结果填入到相应的索引位置
        final_predictions[valid_idx] = y_pred

    # overall_score = roc_auc_score(y, final_predictions)
    overall_score = ks_stat(y, final_predictions)
    print(overall_score)

    # 返回负的AUC分数作为最小化目标 -->  ks
    return {'loss': -overall_score, 'status': STATUS_OK}


# 贝叶斯优化调参函数
def bayesian_optimize_lgbm(X, y, param_space, max_evals=50):
    trials = Trials()

    # 使用fmin函数进行贝叶斯优化
    best_params = fmin(
        fn=lambda params: objective(params, X, y),  # 优化目标
        space=param_space,  # 参数空间
        algo=tpe.suggest,  # 使用TPE算法
        max_evals=max_evals,  # 最大评估次数
        trials=trials,  # 记录每次评估的结果
        rstate=np.random.default_rng(random_state)
    )

    return best_params, trials


def lgb_5_fold(X, y, X_predict, id_predict, best_params, n_folds=5):
    # Map indices back to actual values
    best_params['num_leaves'] = np.arange(20, 200, dtype=int)[best_params['num_leaves']]
    best_params['max_depth'] = np.arange(3, 12, dtype=int)[best_params['max_depth']]

    # Add necessary parameters
    best_params.update({
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'verbose': -1,
        'n_estimators': 10000,
        'metric': 'auc',
        'seed': random_state,
        'bagging_seed': random_state,
        'feature_fraction_seed': random_state,
        'data_random_seed': random_state,
        'deterministic': True,
        'force_col_wise': True
    })

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_scores = []
    final_predictions = np.zeros(len(X))  # 用于存储每一折的预测结果
    prediction_folds_mean = np.zeros(len(X_predict))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # 训练模型
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        model = lgb.train(
            best_params,
            lgb_train,
            feval=lgb_ks_score_eval,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=400),  # Ensure consistency
                lgb.log_evaluation(100)
            ]
        )

        # 保存模型
        # model_path = f"{dir_model}/lgbm_fold_{fold + 1}.bin"
        # model.save_model(model_path)
        # print(f"Model for fold {fold + 1} saved at {model_path}")

        # 验证集预测
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        prediction_folds_mean += (model.predict(X_predict, num_iteration=model.best_iteration) / n_folds)

        # 计算KS或AUC
        score = roc_auc_score(y_valid, y_pred)
        fold_scores.append(score)

        # 将每一折的预测结果保存到对应的索引位置
        final_predictions[valid_idx] = y_pred

        print(f"Fold {fold + 1} - AUC Score: {score:.4f}")

    # 最终综合预测结果的评分
    overall_score = roc_auc_score(y, final_predictions)
    print(f"\nOverall AUC Score: {overall_score:.4f}")

    overall_ks = ks_stat(y, final_predictions)
    print(f"\nOverall KS Score: {overall_ks:.4f}")

    # 提交预测集的概率
    pd.DataFrame({'id': id_predict, 'predicts': prediction_folds_mean}) \
        .to_csv(Path(dir_result) / 'upload.csv', index=False)

    return overall_score


def load_data(dir_path: Path, file_name: str) -> Bunch:
    data = pickle.load(open(dir_path.joinpath(file_name), 'rb'))
    return data


# 读取数据
print("--------- begin load train and predict data ---------")
train_data = load_data(Path(dir_train), 'train.p')
print(f"columns : {train_data.col_names}")
# print(f"category columns : {train_data.category_cols}")
X = train_data.data
y = train_data.target
print(f"X train shape : {X.shape}")
print(f"y train shape : {y.shape}")

test_data = load_data(Path(dir_test), 'test.p')
X_predict = test_data.data
id_predict = test_data.id
print(f"X predict shape : {X_predict.shape}")
print(f"id predict shape : {id_predict.shape}")
print("--------- done load train and predict data ---------")


# 执行贝叶斯优化
best_params, trials = bayesian_optimize_lgbm(X, y, param_space, max_evals=30)
print("Best parameters found:", best_params)

# 执行5折交叉验证
best_score = lgb_5_fold(X, y, X_predict, id_predict,  best_params)
print(f"Best score: {best_score}")
