import pandas as pd

# 假设你已经加载了dataframes: target 和 aps

# 从 target 中抽取测试集
test_target_1 = target[target['FLAG'] == 1].sample(250)
test_target_0 = target[target['FLAG'] == 0].sample(4500)
test_target = pd.concat([test_target_1, test_target_0])

# 从 aps 中抽取测试集
test_aps = aps[aps['CUST_NO'].isin(test_target['CUST_NO'])]

# 删除对应的记录，得到训练集
train_target = target.drop(test_target.index)
train_aps = aps.drop(test_aps.index)
