import os

import pandas as pd

from constant import dir_preprocess
from data import loader, exporter

target = loader.to_concat_df('TARGET_QZ')

flatmap = loader.to_df(os.path.join(dir_preprocess, 'flatmap.csv'))[
    ['CUST_NO', 'NTRL_CUST_AGE', 'NTRL_CUST_SEX_CD', 'DAY_FA_BAL', 'MAVER_FA_BAL', 'SAVER_FA_BAL', 'YAVER_FA_BAL',
     'DAY_AUM_BAL', 'MAVER_AUM_BAL', 'SAVER_AUM_BAL', 'YAVER_AUM_BAL', 'DAY_TOT_IVST_BAL', 'MAVER_TOT_IVST_BAL',
     'SAVER_TOT_IVST_BAL', 'YAVER_TOT_IVST_BAL', 'DAY_DPSA_BAL', 'MAVER_DPSA_BAL', 'SAVER_DPSA_BAL', 'YAVER_DPSA_BAL',
     'DAY_TD_BAL', 'MAVER_TD_BAL', 'SAVER_TD_BAL', 'YAVER_TD_BAL', 'CCARD_IND', 'PAY_IND', 'TDPT_PAY_IND',
     'DAY_TOT_DP_BAL', 'MAVER_TOT_DP_BAL', 'SAVER_TOT_DP_BAL', 'YAVER_TOT_DP_BAL', 'DAY_DPSA_div_MAVER_DPSA',
     'DAY_DPSA_div_SAVER_DPSA', 'DAY_DPSA_div_YAVER_DPSA', 'DAY_DPSA_minus_DAY_TD', 'DAY_DPSA_minus_MAVER_TD',
     'DAY_DPSA_minus_SAVER_TD', 'DAY_DPSA_minus_YAVER_TD', 'DAY_DPSA_minus_DAY_TOT_DP', 'DAY_DPSA_div_MAVER_TOT_DP',
     'DAY_DPSA_div_SAVER_TOT_DP', 'DAY_DPSA_minus_SAVER_TOT_DP', 'DAY_DPSA_div_YAVER_TOT_DP',
     'DAY_DPSA_minus_YAVER_TOT_DP', 'DAY_DPSA_minus_DAY_TOT_IVST', 'DAY_DPSA_minus_MAVER_TOT_IVST',
     'DAY_DPSA_minus_SAVER_TOT_IVST', 'DAY_DPSA_minus_YAVER_TOT_IVST', 'DAY_DPSA_minus_DAY_AUM',
     'DAY_DPSA_div_MAVER_AUM', 'DAY_DPSA_minus_MAVER_AUM', 'DAY_DPSA_div_SAVER_AUM', 'DAY_DPSA_minus_SAVER_AUM',
     'DAY_DPSA_div_YAVER_AUM', 'DAY_DPSA_minus_YAVER_AUM', 'DAY_DPSA_div_DAY_FA', 'DAY_DPSA_minus_DAY_FA',
     'DAY_DPSA_div_MAVER_FA', 'DAY_DPSA_minus_MAVER_FA', 'DAY_DPSA_div_SAVER_FA', 'DAY_DPSA_minus_SAVER_FA',
     'DAY_DPSA_div_YAVER_FA', 'DAY_DPSA_minus_YAVER_FA', 'MAVER_DPSA_div_SAVER_DPSA', 'MAVER_DPSA_div_YAVER_DPSA',
     'MAVER_DPSA_minus_DAY_TD', 'MAVER_DPSA_minus_MAVER_TD', 'MAVER_DPSA_minus_SAVER_TD', 'MAVER_DPSA_minus_YAVER_TD',
     'MAVER_DPSA_div_DAY_TOT_DP', 'MAVER_DPSA_minus_MAVER_TOT_DP', 'MAVER_DPSA_div_SAVER_TOT_DP',
     'MAVER_DPSA_minus_SAVER_TOT_DP', 'MAVER_DPSA_div_YAVER_TOT_DP', 'MAVER_DPSA_minus_YAVER_TOT_DP',
     'MAVER_DPSA_minus_DAY_TOT_IVST', 'MAVER_DPSA_minus_MAVER_TOT_IVST', 'MAVER_DPSA_minus_SAVER_TOT_IVST',
     'MAVER_DPSA_div_DAY_AUM', 'MAVER_DPSA_minus_DAY_AUM', 'MAVER_DPSA_div_MAVER_AUM', 'MAVER_DPSA_minus_MAVER_AUM',
     'MAVER_DPSA_div_SAVER_AUM', 'MAVER_DPSA_minus_SAVER_AUM', 'MAVER_DPSA_div_YAVER_AUM', 'MAVER_DPSA_minus_YAVER_AUM',
     'MAVER_DPSA_div_DAY_FA', 'MAVER_DPSA_minus_DAY_FA', 'MAVER_DPSA_div_MAVER_FA', 'MAVER_DPSA_minus_MAVER_FA',
     'MAVER_DPSA_div_SAVER_FA', 'MAVER_DPSA_minus_SAVER_FA', 'MAVER_DPSA_div_YAVER_FA', 'MAVER_DPSA_minus_YAVER_FA',
     'SAVER_DPSA_div_YAVER_DPSA', 'SAVER_DPSA_minus_YAVER_DPSA', 'SAVER_DPSA_minus_DAY_TD', 'SAVER_DPSA_minus_MAVER_TD',
     'SAVER_DPSA_minus_SAVER_TD', 'SAVER_DPSA_minus_YAVER_TD', 'SAVER_DPSA_div_DAY_TOT_DP',
     'SAVER_DPSA_div_MAVER_TOT_DP', 'SAVER_DPSA_minus_MAVER_TOT_DP', 'SAVER_DPSA_minus_SAVER_TOT_DP',
     'SAVER_DPSA_div_YAVER_TOT_DP', 'SAVER_DPSA_minus_YAVER_TOT_DP', 'SAVER_DPSA_minus_DAY_TOT_IVST',
     'SAVER_DPSA_minus_MAVER_TOT_IVST', 'SAVER_DPSA_minus_SAVER_TOT_IVST', 'SAVER_DPSA_minus_YAVER_TOT_IVST',
     'SAVER_DPSA_div_DAY_AUM', 'SAVER_DPSA_minus_DAY_AUM', 'SAVER_DPSA_div_MAVER_AUM', 'SAVER_DPSA_minus_MAVER_AUM',
     'SAVER_DPSA_div_SAVER_AUM', 'SAVER_DPSA_minus_SAVER_AUM', 'SAVER_DPSA_div_YAVER_AUM', 'SAVER_DPSA_minus_YAVER_AUM',
     'SAVER_DPSA_div_DAY_FA', 'SAVER_DPSA_div_MAVER_FA', 'SAVER_DPSA_minus_MAVER_FA', 'SAVER_DPSA_div_SAVER_FA',
     'SAVER_DPSA_minus_SAVER_FA', 'SAVER_DPSA_div_YAVER_FA', 'SAVER_DPSA_minus_YAVER_FA', 'YAVER_DPSA_minus_DAY_TD',
     'YAVER_DPSA_minus_MAVER_TD', 'YAVER_DPSA_minus_SAVER_TD', 'YAVER_DPSA_minus_YAVER_TD', 'YAVER_DPSA_div_DAY_TOT_DP',
     'YAVER_DPSA_div_MAVER_TOT_DP', 'YAVER_DPSA_div_SAVER_TOT_DP', 'YAVER_DPSA_minus_SAVER_TOT_DP',
     'YAVER_DPSA_minus_YAVER_TOT_DP', 'YAVER_DPSA_minus_DAY_TOT_IVST', 'YAVER_DPSA_minus_MAVER_TOT_IVST',
     'YAVER_DPSA_minus_SAVER_TOT_IVST', 'YAVER_DPSA_minus_YAVER_TOT_IVST', 'YAVER_DPSA_div_DAY_AUM',
     'YAVER_DPSA_div_MAVER_AUM', 'YAVER_DPSA_minus_MAVER_AUM', 'YAVER_DPSA_div_SAVER_AUM', 'YAVER_DPSA_minus_SAVER_AUM',
     'YAVER_DPSA_div_YAVER_AUM', 'YAVER_DPSA_minus_YAVER_AUM', 'YAVER_DPSA_div_DAY_FA', 'YAVER_DPSA_div_MAVER_FA',
     'YAVER_DPSA_div_SAVER_FA', 'YAVER_DPSA_div_YAVER_FA', 'YAVER_DPSA_minus_YAVER_FA', 'DAY_TD_minus_DAY_TOT_DP',
     'DAY_TD_div_MAVER_TOT_DP', 'DAY_TD_minus_MAVER_TOT_DP', 'DAY_TD_minus_SAVER_TOT_DP', 'DAY_TD_minus_YAVER_TOT_DP',
     'DAY_TD_minus_DAY_TOT_IVST', 'DAY_TD_minus_MAVER_TOT_IVST', 'DAY_TD_minus_SAVER_TOT_IVST',
     'DAY_TD_minus_YAVER_TOT_IVST', 'DAY_TD_minus_DAY_AUM', 'DAY_TD_minus_MAVER_AUM', 'DAY_TD_minus_SAVER_AUM',
     'DAY_TD_minus_YAVER_AUM', 'DAY_TD_minus_DAY_FA', 'DAY_TD_div_MAVER_FA', 'DAY_TD_minus_MAVER_FA',
     'DAY_TD_minus_SAVER_FA', 'DAY_TD_div_YAVER_FA', 'DAY_TD_minus_YAVER_FA', 'MAVER_TD_minus_DAY_TOT_DP',
     'MAVER_TD_minus_MAVER_TOT_DP', 'MAVER_TD_minus_SAVER_TOT_DP', 'MAVER_TD_minus_YAVER_TOT_DP',
     'MAVER_TD_minus_DAY_TOT_IVST', 'MAVER_TD_minus_MAVER_TOT_IVST', 'MAVER_TD_minus_SAVER_TOT_IVST',
     'MAVER_TD_minus_YAVER_TOT_IVST', 'MAVER_TD_minus_DAY_AUM', 'MAVER_TD_minus_MAVER_AUM', 'MAVER_TD_minus_SAVER_AUM',
     'MAVER_TD_minus_YAVER_AUM', 'MAVER_TD_minus_DAY_FA', 'MAVER_TD_minus_MAVER_FA', 'MAVER_TD_minus_SAVER_FA',
     'MAVER_TD_minus_YAVER_FA', 'SAVER_TD_minus_DAY_TOT_DP', 'SAVER_TD_minus_MAVER_TOT_DP',
     'SAVER_TD_minus_SAVER_TOT_DP', 'SAVER_TD_minus_YAVER_TOT_DP', 'SAVER_TD_minus_DAY_TOT_IVST',
     'SAVER_TD_minus_MAVER_TOT_IVST', 'SAVER_TD_minus_SAVER_TOT_IVST', 'SAVER_TD_minus_YAVER_TOT_IVST',
     'SAVER_TD_minus_DAY_AUM', 'SAVER_TD_minus_MAVER_AUM', 'SAVER_TD_minus_SAVER_AUM', 'SAVER_TD_minus_YAVER_AUM',
     'SAVER_TD_minus_DAY_FA', 'SAVER_TD_minus_MAVER_FA', 'SAVER_TD_minus_SAVER_FA', 'SAVER_TD_minus_YAVER_FA',
     'YAVER_TD_minus_DAY_TOT_DP', 'YAVER_TD_minus_MAVER_TOT_DP', 'YAVER_TD_minus_SAVER_TOT_DP',
     'YAVER_TD_minus_YAVER_TOT_DP', 'YAVER_TD_minus_DAY_TOT_IVST', 'YAVER_TD_minus_MAVER_TOT_IVST',
     'YAVER_TD_minus_SAVER_TOT_IVST', 'YAVER_TD_minus_YAVER_TOT_IVST', 'YAVER_TD_minus_DAY_AUM',
     'YAVER_TD_minus_MAVER_AUM', 'YAVER_TD_minus_SAVER_AUM', 'YAVER_TD_minus_YAVER_AUM', 'YAVER_TD_minus_DAY_FA',
     'YAVER_TD_minus_MAVER_FA', 'YAVER_TD_minus_SAVER_FA', 'YAVER_TD_minus_YAVER_FA', 'DAY_TOT_DP_div_MAVER_TOT_DP',
     'DAY_TOT_DP_div_SAVER_TOT_DP', 'DAY_TOT_DP_div_YAVER_TOT_DP', 'DAY_TOT_DP_minus_DAY_TOT_IVST',
     'DAY_TOT_DP_minus_MAVER_TOT_IVST', 'DAY_TOT_DP_minus_SAVER_TOT_IVST', 'DAY_TOT_DP_minus_YAVER_TOT_IVST',
     'DAY_TOT_DP_minus_DAY_AUM', 'DAY_TOT_DP_div_MAVER_AUM', 'DAY_TOT_DP_div_SAVER_AUM', 'DAY_TOT_DP_div_YAVER_AUM',
     'DAY_TOT_DP_minus_DAY_FA', 'DAY_TOT_DP_div_MAVER_FA', 'DAY_TOT_DP_div_SAVER_FA', 'DAY_TOT_DP_div_YAVER_FA',
     'MAVER_TOT_DP_div_SAVER_TOT_DP', 'MAVER_TOT_DP_div_YAVER_TOT_DP', 'MAVER_TOT_DP_minus_DAY_TOT_IVST',
     'MAVER_TOT_DP_minus_MAVER_TOT_IVST', 'MAVER_TOT_DP_minus_SAVER_TOT_IVST', 'MAVER_TOT_DP_minus_YAVER_TOT_IVST',
     'MAVER_TOT_DP_div_DAY_AUM', 'MAVER_TOT_DP_minus_MAVER_AUM', 'MAVER_TOT_DP_div_SAVER_AUM',
     'MAVER_TOT_DP_minus_SAVER_AUM', 'MAVER_TOT_DP_div_YAVER_AUM', 'MAVER_TOT_DP_div_DAY_FA',
     'MAVER_TOT_DP_div_SAVER_FA', 'MAVER_TOT_DP_minus_SAVER_FA', 'MAVER_TOT_DP_div_YAVER_FA',
     'MAVER_TOT_DP_minus_YAVER_FA', 'SAVER_TOT_DP_div_YAVER_TOT_DP', 'SAVER_TOT_DP_minus_DAY_TOT_IVST',
     'SAVER_TOT_DP_minus_MAVER_TOT_IVST', 'SAVER_TOT_DP_minus_SAVER_TOT_IVST', 'SAVER_TOT_DP_minus_YAVER_TOT_IVST',
     'SAVER_TOT_DP_div_DAY_AUM', 'SAVER_TOT_DP_div_MAVER_AUM', 'SAVER_TOT_DP_minus_MAVER_AUM',
     'SAVER_TOT_DP_minus_SAVER_AUM', 'SAVER_TOT_DP_div_YAVER_AUM', 'SAVER_TOT_DP_minus_YAVER_AUM',
     'SAVER_TOT_DP_div_DAY_FA', 'SAVER_TOT_DP_div_MAVER_FA', 'SAVER_TOT_DP_div_YAVER_FA', 'SAVER_TOT_DP_minus_YAVER_FA',
     'YAVER_TOT_DP_minus_DAY_TOT_IVST', 'YAVER_TOT_DP_minus_MAVER_TOT_IVST', 'YAVER_TOT_DP_minus_SAVER_TOT_IVST',
     'YAVER_TOT_DP_minus_YAVER_TOT_IVST', 'YAVER_TOT_DP_div_DAY_AUM', 'YAVER_TOT_DP_div_MAVER_AUM',
     'YAVER_TOT_DP_div_SAVER_AUM', 'YAVER_TOT_DP_minus_SAVER_AUM', 'YAVER_TOT_DP_minus_YAVER_AUM',
     'YAVER_TOT_DP_div_DAY_FA', 'YAVER_TOT_DP_div_MAVER_FA', 'YAVER_TOT_DP_div_SAVER_FA', 'YAVER_TOT_DP_minus_YAVER_FA',
     'DAY_TOT_IVST_minus_DAY_AUM', 'DAY_TOT_IVST_minus_MAVER_AUM', 'DAY_TOT_IVST_minus_SAVER_AUM',
     'DAY_TOT_IVST_minus_YAVER_AUM', 'DAY_TOT_IVST_minus_DAY_FA', 'DAY_TOT_IVST_minus_MAVER_FA',
     'DAY_TOT_IVST_minus_SAVER_FA', 'DAY_TOT_IVST_minus_YAVER_FA', 'MAVER_TOT_IVST_minus_DAY_AUM',
     'MAVER_TOT_IVST_minus_MAVER_AUM', 'MAVER_TOT_IVST_minus_SAVER_AUM', 'MAVER_TOT_IVST_minus_YAVER_AUM',
     'MAVER_TOT_IVST_minus_DAY_FA', 'MAVER_TOT_IVST_minus_MAVER_FA', 'MAVER_TOT_IVST_minus_SAVER_FA',
     'MAVER_TOT_IVST_minus_YAVER_FA', 'SAVER_TOT_IVST_minus_DAY_AUM', 'SAVER_TOT_IVST_minus_MAVER_AUM',
     'SAVER_TOT_IVST_minus_SAVER_AUM', 'SAVER_TOT_IVST_minus_YAVER_AUM', 'SAVER_TOT_IVST_minus_DAY_FA',
     'SAVER_TOT_IVST_minus_MAVER_FA', 'SAVER_TOT_IVST_minus_SAVER_FA', 'SAVER_TOT_IVST_minus_YAVER_FA',
     'YAVER_TOT_IVST_minus_DAY_AUM', 'YAVER_TOT_IVST_minus_MAVER_AUM', 'YAVER_TOT_IVST_minus_SAVER_AUM',
     'YAVER_TOT_IVST_minus_YAVER_AUM', 'YAVER_TOT_IVST_minus_DAY_FA', 'YAVER_TOT_IVST_minus_MAVER_FA',
     'YAVER_TOT_IVST_minus_SAVER_FA', 'YAVER_TOT_IVST_minus_YAVER_FA', 'DAY_AUM_div_MAVER_AUM', 'DAY_AUM_div_SAVER_AUM',
     'DAY_AUM_div_YAVER_AUM', 'DAY_AUM_div_MAVER_FA', 'DAY_AUM_div_SAVER_FA', 'DAY_AUM_div_YAVER_FA',
     'MAVER_AUM_div_SAVER_AUM', 'MAVER_AUM_minus_SAVER_AUM', 'MAVER_AUM_div_YAVER_AUM', 'MAVER_AUM_div_DAY_FA',
     'MAVER_AUM_div_SAVER_FA', 'MAVER_AUM_minus_SAVER_FA', 'MAVER_AUM_div_YAVER_FA', 'SAVER_AUM_div_YAVER_AUM',
     'SAVER_AUM_div_DAY_FA', 'SAVER_AUM_div_MAVER_FA', 'SAVER_AUM_div_YAVER_FA', 'SAVER_AUM_minus_YAVER_FA',
     'YAVER_AUM_div_DAY_FA', 'YAVER_AUM_div_MAVER_FA', 'YAVER_AUM_div_SAVER_FA', 'DAY_FA_div_MAVER_FA',
     'DAY_FA_div_SAVER_FA', 'DAY_FA_div_YAVER_FA', 'MAVER_FA_div_SAVER_FA', 'MAVER_FA_div_YAVER_FA',
     'SAVER_FA_div_YAVER_FA']]
g_stat = loader.to_df(os.path.join(dir_preprocess, 'g_stat.csv'))
time_part0 = loader.to_df(os.path.join(dir_preprocess, 'time_series_part1.csv'))
time_part2 = loader.to_df(os.path.join(dir_preprocess, 'time_series_part2.csv'))

merged = target.merge(flatmap, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left') \
    .merge(g_stat, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left') \
    .merge(time_part0, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left') \
    .merge(time_part2, left_on=['CARD_NO'], right_on=['CRD_SRC'], how='left')

merged = merged.drop(columns=['CRD_SRC', 'DATA_DAT', 'CARD_NO'])

print(merged.describe())

train = merged[merged['SRC'] == 'train']
train = train.drop(columns=['SRC'])

test = merged[merged['SRC'] == 'test']
test = test.drop(columns=['SRC', 'FLAG'])

exporter.export_df_to_preprocess('train', train)
exporter.export_df_to_preprocess('test', test)
