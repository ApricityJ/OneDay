import os
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from constant import dir_preprocess
from data import loader, exporter

target = loader.to_concat_df('TARGET')

flatmap = loader.to_df(Path(dir_preprocess).joinpath(f'flatmap.csv'))
time_part1 = loader.to_df(Path(dir_preprocess).joinpath(f'time_series_part1.csv'))
time_part2 = loader.to_df(Path(dir_preprocess).joinpath(f'time_series_part2.csv'))
set_part1 = loader.to_df(Path(dir_preprocess).joinpath(f'set_part1.csv'))
set_part3 = loader.to_df(Path(dir_preprocess).joinpath(f'set_part3.csv'))
g_stat = loader.to_df(Path(dir_preprocess).joinpath(f'g_stat.csv'))
freewill = loader.to_df(Path(dir_preprocess).joinpath(f'freewill.csv'))
sim_part1 = loader.to_df(Path(dir_preprocess).joinpath(f'sim_part1.csv'))

merged = target.merge(flatmap, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left') \
    .merge(g_stat, left_on=['CARD_NO'], right_on=['node'], how='left') \
    .merge(time_part1, left_on=['CARD_NO'], right_on=['CRD_SRC'], how='left') \
    .merge(time_part2, left_on=['CARD_NO'], right_on=['CRD_SRC'], how='left') \
    .merge(set_part1, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left') \
    .merge(set_part3, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left') \
    .merge(freewill, left_on=['CARD_NO'], right_on=['CRD_SRC'], how='left') \
    .merge(sim_part1, left_on=['CUST_NO'], right_on=['CUST_NO'], how='left')


merged = merged.drop(columns=['CRD_SRC', 'DATA_DAT', 'CARD_NO', 'node', 'CRD_SRC_x', 'CRD_SRC_y'])

print(merged.describe())
print(merged.columns)

train = merged[merged['SRC_x'] == 'train']
train = train.drop(columns=['SRC_x', 'SRC_y'])

test = merged[merged['SRC_x'] == 'test']
test = test.drop(columns=['SRC_x', 'SRC_y', 'FLAG'])

exporter.export_df_to_preprocess('train', train)
exporter.export_df_to_preprocess('test', test)
