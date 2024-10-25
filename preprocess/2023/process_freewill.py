import os

import pandas as pd
from scipy.stats import entropy

from constant import dir_preprocess
from data import loader, exporter

df = loader.to_df(os.path.join(dir_preprocess, 'APS.csv'))


# 等金额的收-付配对交易的数量和交易流水
def equal_amount_pairs(card_group):
    positive_transactions = card_group[card_group['TRN_AMT'] > 0]['TRN_AMT'].tolist()
    negative_transactions = card_group[card_group['TRN_AMT'] < 0]['TRN_AMT'].tolist()

    paired_transactions = 0
    paired_amount = 0
    for amount in positive_transactions:
        if -amount in negative_transactions:
            paired_transactions += 1
            paired_amount += abs(amount)
            negative_transactions.remove(-amount)

    return pd.Series([paired_transactions, paired_amount])


equal_pairs_df = df.groupby('CRD_SRC').apply(equal_amount_pairs).reset_index()
equal_pairs_df.columns = ['CRD_SRC', 'Equal_Trans_Count', 'Equal_Trans_Amount']


def compute_entropy(card_group):
    value_counts = card_group['CRD_TGT'].value_counts(normalize=True)
    return entropy(value_counts)


entropy_df = df.groupby('CRD_SRC').apply(compute_entropy).reset_index()
entropy_df.columns = ['CRD_SRC', 'Shannon_Entropy']


def multi_transaction_pattern(card_group):
    transaction_series = card_group['TRN_AMT'].tolist()
    matched_series_count = 0
    matched_amount = 0
    for i in range(len(transaction_series)):
        for j in range(i + 2, min(i + 8, len(transaction_series) + 1)):
            if abs(sum(transaction_series[i:j])) < 0.01:
                matched_series_count += 1
                matched_amount += sum([abs(x) for x in transaction_series[i:j]])

    return pd.Series([matched_series_count, matched_amount])


multi_trans_df = df.groupby('CRD_SRC').apply(multi_transaction_pattern).reset_index()
multi_trans_df.columns = ['CRD_SRC', 'Multi_Trans_Count', 'Multi_Trans_Amount']

total_trans = df.groupby('CRD_SRC').agg(Total_Trans_Count=('TRN_DT', 'size'),
                                        Total_Trans_Amount=('TRN_AMT', 'sum')).reset_index()

merged_df = pd.merge(total_trans, equal_pairs_df, on='CRD_SRC')
merged_df = pd.merge(merged_df, entropy_df, on='CRD_SRC')
merged_df = pd.merge(merged_df, multi_trans_df, on='CRD_SRC')

# Calculate ratios
merged_df['Equal_Trans_Count_Ratio'] = merged_df['Equal_Trans_Count'] / merged_df['Total_Trans_Count']
merged_df['Equal_Trans_Amount_Ratio'] = merged_df['Equal_Trans_Amount'] / merged_df['Total_Trans_Amount']
merged_df['Multi_Trans_Count_Ratio'] = merged_df['Multi_Trans_Count'] / merged_df['Total_Trans_Count']
merged_df['Multi_Trans_Amount_Ratio'] = merged_df['Multi_Trans_Amount'] / merged_df['Total_Trans_Amount']

exporter.export_df_to_preprocess('freewill', merged_df)
