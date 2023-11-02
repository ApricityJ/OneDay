import pandas as pd
from sklearn.model_selection import KFold
from constant import active_random_state



# 创建示例数据
data = pd.DataFrame({
    'X': ['A', 'B', 'A', 'A', 'B', 'C', 'C', 'A', 'B', 'C'],
    'y': [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
})

# 设置K折交叉验证和平滑参数
kf = KFold(n_splits=5, shuffle=True, random_state=active_random_state)
alpha = 2.0  # 平滑参数
global_mean = data['y'].mean()

# 初始化一个新列来存储目标编码的结果
data['X_encoded'] = 0

# 对每一折进行目标编码
for tr_ind, val_ind in kf.split(data):
    X_tr, X_val = data.loc[tr_ind], data.loc[val_ind]

    # 使用训练数据计算目标编码
    encoding = (X_tr.groupby('X')['y'].sum() + global_mean * alpha) / (X_tr.groupby('X')['y'].count() + alpha)

    # 应用目标编码到验证数据
    data.loc[val_ind, 'X_encoded'] = X_val['X'].map(encoding).fillna(global_mean)

print(data)
