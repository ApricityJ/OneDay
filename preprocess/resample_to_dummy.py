import pandas as pd
from data import loader, exporter
from constant import *

ds_train = {}
ds_test = {}

key_maps = {
    'MBANK_TRNFLW_QZ': {'src': 'CUST_NO', 'map': 'TFT_CSTNO'},
    'MBANK_QRYTRNFLW_QZ': {'src': 'CUST_NO', 'map': 'TFT_CSTNO'},
    'EBANK_CSTLOG_QZ': {'src': 'CUST_NO', 'map': 'CSTNO'},
    'EBANK_CSTLOGQUERY_QZ': {'src': 'CUST_NO', 'map': 'CLQ_CSTNO'},
    'APS_QZ': {'src': 'CARD_NO', 'map': 'APSDPRDNO'},
    'CUST_FA_SUM_QZ': {'src': 'CUST_NO', 'map': 'CUST_NO'},
    'DP_CUST_SUM_QZ': {'src': 'CUST_NO', 'map': 'CUST_NO'},
    'TAGS_PROD_HOLD_QZ': {'src': 'CUST_NO', 'map': 'CUST_NO'},
    'NATURE_CUST_QZ': {'src': 'CUST_NO', 'map': 'CUST_NO'}
}

ds = loader.to_ds_train()

# 从 target 中抽取测试集
target = ds['TARGET_QZ']
test_target_1 = target[target['FLAG'] == 1].sample(250)
test_target_0 = target[target['FLAG'] == 0].sample(4500)
test_target = pd.concat([test_target_1, test_target_0])
train_target = target.drop(test_target.index)

ds_train['TARGET'] = train_target
ds_test['TARGET'] = test_target

# 抽取测试集
others = [key for key in ds.keys() if key != 'TARGET_QZ']
for key in others:
    # 提取
    df = ds[key]
    key_map = key_maps[key]

    # 筛选
    df_test = df[df[key_map['map']].isin(test_target[key_map['src']])]
    df_train = df.drop(df_test.index)

    # 收集
    key_new = key[:-3]
    ds_train[key_new] = df_train
    ds_test[key_new] = df_test

# 保存
p_train = os.path.join(dir_base, active_year, 'dummy', 'train')
p_test = os.path.join(dir_base, active_year, 'dummy', 'test')

exporter.export_ds(p_train, ds_train)
exporter.export_ds(p_test, ds_test)
