from data import loader, exporter
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count, Array
from scipy import stats

# part 1 完成一个基本信息的相似度



target = loader.to_concat_df('TARGET')
df_na = loader.to_concat_df('NATURE_CUST')
df_fa = loader.to_concat_df('CUST_FA_SUM')
df_dp = loader.to_concat_df('DP_CUST_SUM')
df_pd = loader.to_concat_df('TAGS_PROD_HOLD')

# 关联切片表
df = target \
    .merge(df_na, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left', suffixes=('', '_NA')) \
    .merge(df_fa, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left', suffixes=('', '_FA')) \
    .merge(df_dp, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left', suffixes=('', '_DP')) \
    .merge(df_pd, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left', suffixes=('', '_PD'))

sex_map = {'A': 1, 'B': 0}
rank_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

# 使用 replace() 函数替换值
df['NTRL_CUST_SEX_CD'] = df['NTRL_CUST_SEX_CD'].replace(sex_map)
df['NTRL_RANK_CD'] = df['NTRL_RANK_CD'].replace(rank_map)

# 删除重复列
df.drop(columns=['DATA_DAT', 'SRC', 'DATA_DAT_FA', 'DATA_DAT_DP', 'DATA_DAT_PD', 'CARD_NO'], inplace=True)

# 0. 数据预处理
df.fillna(0, inplace=True)
features = df.columns[2:]  # COL1, COL2,...,COLN
df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
df.fillna(0, inplace=True)

# 1. 按FLAG拆分数据并转为数组
df_fraud_values = df[df['FLAG'] == 1][features].values
df_values = df[features].values
cust_no = df['CUST_NO'].values

# 2. 定义计算余弦相似度的函数
def compute_similarity(start_end):
    start, end = start_end
    sub_result = []
    for index in range(start, end):
        similarities = cosine_similarity(df_values[index].reshape(1, -1), df_fraud_values).flatten()
        similarities = np.round(similarities, 3)
        max_sim = np.max(similarities)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # 计算满足条件的相似度的数量
        count_gt_07 = np.sum(similarities > 0.7)
        count_gt_08 = np.sum(similarities > 0.8)
        count_gt_09 = np.sum(similarities > 0.9)
        count_gt_095 = np.sum(similarities > 0.95)

        # 计算满足条件的相似度的比例
        ratio_gt_07 = count_gt_07 / 1950.0
        ratio_gt_08 = count_gt_08 / 1950.0
        ratio_gt_09 = count_gt_09 / 1950.0
        ratio_gt_095 = count_gt_095 / 1950.0
        sub_result.append((cust_no[index], max_sim, mean_sim, std_sim,count_gt_07, ratio_gt_07, count_gt_08, ratio_gt_08, count_gt_09, ratio_gt_09, count_gt_095, ratio_gt_095))
    return sub_result


if __name__ == '__main__':
    # 将数据分为小块
    chunk_size = len(df) // cpu_count()
    indices = [(i, i + chunk_size) for i in range(0, len(df), chunk_size)]

    # 调整最后一个块的结束索引
    if indices[-1][1] > len(df):
        indices[-1] = (indices[-1][0], len(df))

    # 使用多进程计算
    pool = Pool(processes=cpu_count())
    results = pool.map(compute_similarity, indices)
    pool.close()
    pool.join()

    # 将结果汇总并转换为DataFrame
    results_flat = [item for sublist in results for item in sublist]
    result_df = pd.DataFrame(results_flat, columns=["CUST_NO", "MAX_SIM", "MEAN_SIM", "STD_SIM", 'COUNT_GT_07', 'RATIO_GT_07', 'COUNT_GT_08', 'RATIO_GT_08', 'COUNT_GT_09', 'RATIO_GT_09', 'COUNT_GT_095', 'RATIO_GT_095'])

    exporter.export_df_to_preprocess('sim_part1', result_df)