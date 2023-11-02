import numpy as np

from utils import loader, exporter

# 基础表
df_cust = loader.to_concat_df('NATURE_CUST_QZ')
df_fa = loader.to_concat_df('CUST_FA_SUM_QZ')
df_dp = loader.to_concat_df('DP_CUST_SUM_QZ')
df_pd = loader.to_concat_df('TAGS_PROD_HOLD_QZ')

# 关联切片表
df_flat = df_cust \
    .merge(df_fa, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left', suffixes=('', '_FA')) \
    .merge(df_dp, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left', suffixes=('', '_DP')) \
    .merge(df_pd, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left', suffixes=('', '_PD'))

# 删除重复列
df_flat.drop(columns=['DATA_DAT_DP', 'DATA_DAT_PD'], inplace=True)

# 规范一下列名
df_flat = df_flat.rename(
    columns={'TOT_IVST_BAL': 'DAY_TOT_IVST_BAL', 'DPSA_BAL': 'DAY_DPSA_BAL', 'TD_BAL': 'DAY_TD_BAL'})

# 对空值填充0
df_flat = df_flat.fillna(0)

# 存款总数
periods = ['DAY', 'MAVER', 'SAVER', 'YAVER']
for p in periods:
    df_flat[f'{p}_TOT_DP_BAL'] = df_flat[f'{p}_DPSA_BAL'] + df_flat[f'{p}_TD_BAL']

# 产品维度
measures = ['DPSA', 'TD', 'TOT_DP', 'TOT_IVST', 'AUM', 'FA']

# 产品*时间 展开后的维度
dims = [f'{p}_{m}_BAL' for m in measures for p in periods]

# 维度间运算，两两相减和除
for i in range(len(dims)):
    A = dims[i]
    for B in dims[i + 1:]:
        C = '{0}_div_{1}'.format(A.replace('_BAL', ''), B.replace('_BAL', ''))
        D = '{0}_minus_{1}'.format(A.replace('_BAL', ''), B.replace('_BAL', ''))
        df_flat[C] = np.where(df_flat[B].notna() & (df_flat[B] != 0), round(df_flat[A] / df_flat[B], 3), np.nan)
        df_flat[D] = np.where(df_flat[A].notna() & df_flat[B].notna(), round(df_flat[A] - df_flat[B], 3), np.nan)

# 保存至 preprocess文件夹，命名为flatmap.csv
exporter.export_df_to_preprocess('flatmap', df_flat)
