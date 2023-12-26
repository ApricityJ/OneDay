# 集合特征
import json

import pandas as pd
from data import exporter, loader


def to_trn_code_diff(dataset, count_threshold, diff_threshold):
    # 1. 统计总涉诈交易和总正常交易的数量
    total_fraud_count = dataset[dataset['FLAG'] == 1].shape[0]
    total_normal_count = dataset[dataset['FLAG'] == 0].shape[0]

    # 统计涉诈卡（CRD_SRC）与正常卡（CRD_SRC）的数量，注意统计卡的数量的时候需要去重复
    total_fraud_cards = len(dataset[dataset['FLAG'] == 1]['CRD_SRC'].unique())
    total_normal_cards = len(dataset[dataset['FLAG'] == 0]['CRD_SRC'].unique())

    # 2. 按交易码分组，汇总涉诈交易和正常交易的数量
    grouped = dataset.groupby(['TRN_COD', 'FLAG'])
    trn_code_stats = grouped.size().unstack(fill_value=0).reset_index()
    trn_code_stats.columns = ['TRN_COD', 'normal_count', 'fraud_count']
    trn_code_stats['total_count'] = trn_code_stats['normal_count'] + trn_code_stats['fraud_count']

    # 过滤数量阈值
    trn_code_stats = trn_code_stats[trn_code_stats['total_count'] > count_threshold]

    # 统计涉诈卡（CRD_SRC）与正常卡（CRD_SRC）的数量
    fraud_cards = grouped['CRD_SRC'].nunique().unstack(fill_value=0)[1].reset_index(drop=True)
    normal_cards = grouped['CRD_SRC'].nunique().unstack(fill_value=0)[0].reset_index(drop=True)
    trn_code_stats['fraud_cards'] = fraud_cards
    trn_code_stats['normal_cards'] = normal_cards

    # 3. 使用2与1比较，得到该交易码中涉诈交易比例与正常交易比例以及卡数量的比例
    trn_code_stats['fraud_ratio'] = trn_code_stats['fraud_count'] / total_fraud_count
    trn_code_stats['normal_ratio'] = trn_code_stats['normal_count'] / total_normal_count
    trn_code_stats['fraud_card_ratio'] = trn_code_stats['fraud_cards'] / total_fraud_cards
    trn_code_stats['normal_card_ratio'] = trn_code_stats['normal_cards'] / total_normal_cards

    # 4. 返回差异大于差异阈值的交易码
    trn_code_stats['fraud_diff'] = (trn_code_stats['fraud_ratio'] - trn_code_stats['normal_ratio']).abs()
    trn_code_stats['card_diff'] = (trn_code_stats['fraud_card_ratio'] - trn_code_stats['normal_card_ratio']).abs()
    trn_code_stats = trn_code_stats[
        (trn_code_stats['fraud_diff'] > diff_threshold) | (trn_code_stats['card_diff'] > diff_threshold)]

    # 输出为字典并保留三位小数
    trn_code_stats = trn_code_stats.round(3)
    result = trn_code_stats.set_index('TRN_COD').T.to_dict()

    return result


def compute_feature_trn_code(dataset, feature_trn_codes):
    # 按CRD_SRC分组，得到每个CRD_SRC的交易码列表
    grouped = dataset.groupby('CRD_SRC')

    # 获取每个CRD_SRC的交易码列表，并重置索引
    trn_code_list_per_crd_src = grouped['TRN_COD'].apply(list).reset_index(name="TRN_CODES")

    # 初始化一个新的dataframe用于保存结果
    result_df = pd.DataFrame()
    result_df['CRD_SRC'] = trn_code_list_per_crd_src['CRD_SRC']

    # 首先，为整个数据集计算每个交易码的平均交易次数和交易占比
    total_trn_codes = dataset['TRN_COD'].tolist()
    avg_counts = {}
    avg_ratios = {}

    for code in feature_trn_codes:
        avg_counts[code] = total_trn_codes.count(code) / len(result_df)
        avg_ratios[code] = avg_counts[code] / len(total_trn_codes)

    # 对于每个特征交易码，计算其在每个CRD_SRC的交易码列表中的次数、占比以及与平均值的差值
    for code in feature_trn_codes:
        current_count = trn_code_list_per_crd_src['TRN_CODES'].apply(lambda x: x.count(code))
        current_ratio = current_count / trn_code_list_per_crd_src['TRN_CODES'].apply(
            lambda x: len(x) if len(x) > 0 else 1)

        result_df[f'{code}_count'] = current_count
        result_df[f'{code}_ratio'] = current_ratio
        result_df[f'{code}_count_diff_from_avg'] = current_count - avg_counts[code]
        result_df[f'{code}_ratio_diff_from_avg'] = current_ratio - avg_ratios[code]

    return result_df.round(3)


def refactor_df(key: str, df):
    column_map = {
        'MBANK_QRYTRNFLW': ['TFT_CSTNO', 'TFT_DTE_TIME', 'TFT_STDBSNCOD'],
        'EBANK_CSTLOGQUERY': ['CLQ_CSTNO', 'CLQ_DTE_TIME', 'CLQ_BSNCOD']
    }
    df = df[column_map[key]]
    df.columns = ['CRD_SRC', 'TRN_DT', 'TRN_COD']
    return df


def to_train_fact(key: str):
    fact, _ = loader.to_df_train_test(key)
    fact = refactor_df(key, fact)
    columns = fact.columns.to_list() + ['FLAG']
    target, _ = loader.to_df_train_test('TARGET')
    fact = fact.merge(target, left_on=['CRD_SRC'], right_on=['CUST_NO'], how='left')
    return fact[columns]


def to_set_features_part3():
    params = {
        'MBANK_QRYTRNFLW': {
            'threshold_count': 1000,
            'threshold_diff': 0.1
        },
        'EBANK_CSTLOGQUERY': {
            'threshold_count': 1000,
            'threshold_diff': 0.1
        }
    }

    target = loader.to_concat_df('TARGET')

    for key in params.keys():
        # 拼接train，获取flag用于训练
        train = to_train_fact(key)
        # concat train和test
        df = loader.to_concat_df(key)
        df = refactor_df(key, df)
        # 根据train得到交易码
        trn_codes = to_trn_code_diff(train, params[key]['threshold_count'], params[key]['threshold_diff']).keys()
        print(trn_codes)
        # 对 train和test生成特征
        df_set_features = compute_feature_trn_code(df, trn_codes)
        # 合并3张表的特征
        target = target.merge(df_set_features, left_on=['CUST_NO'], right_on=['CRD_SRC'], how='left')
        target = target.drop(columns=['CRD_SRC'])

    target = target.drop(columns=['CARD_NO', 'SRC', 'FLAG','DATA_DAT'])
    exporter.export_df_to_preprocess('set_part3', target)


if __name__ == '__main__':
    to_set_features_part3()
