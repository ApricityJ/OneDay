import os.path
from multiprocessing import cpu_count, Pool
import time
import warnings

import pandas as pd

from constant import dir_preprocess
from util import loader, exporter

warnings.filterwarnings("ignore")

# 定义需要处理的时间段
# 如果继续细分，将生成大于样本数量的特征
# time_periods = ['10s', '30s', '1T', '3T', '10T', '1H', '6H', '12H', '1D', '3D', '7D', '14D', '30D']
# time_periods = [None, '1T', '1H', '1D']
time_periods = ['1T']


# 放平心态 - 精细化留待以后
def to_aps():
    df = loader.to_df(os.path.join(dir_preprocess, 'APS.csv'))
    df['TRN_DT'] = pd.to_datetime(df['TRN_DT'], format='%Y%m%d%H%M%S')
    return df


# 是否存在一个序列，相加后近似为0
def check_subsequence(df):
    amounts = df['TRN_AMT'].tolist()

    valid_subseq_count = 0
    total_trans_in_subseq = 0
    total_amt_in_subseq = 0

    for i in range(len(amounts)):
        for j in range(i + 1, min(i + 6, len(amounts) + 1)):  # 加入限制条件，使子序列的最大长度为5
            subseq_sum = sum(amounts[i:j])
            if -0.1 <= subseq_sum <= 0.1:
                valid_subseq_count += 1
                total_trans_in_subseq += len(amounts[i:j])
                total_amt_in_subseq += subseq_sum

    trans_ratio = total_trans_in_subseq / len(df) if len(df) > 0 else 0
    amt_ratio = total_amt_in_subseq / df['TRN_AMT'].sum() if df['TRN_AMT'].sum() != 0 else 0

    return valid_subseq_count, trans_ratio, amt_ratio


# 金额一致的次数
def check_absolute_amount(df):
    total_count = len(df)
    absolute_counts = df['TRN_AMT'].abs().value_counts()
    absolute_same_count = sum(count for amount, count in absolute_counts.items() if count > 1)
    absolute_same_ratio = absolute_same_count / total_count if total_count > 0 else 0
    return absolute_same_count, absolute_same_ratio


# 相似金额出现的次数
def check_similar_amount(df):
    amounts = df['TRN_AMT'].tolist()

    similar_count = 0

    for i in range(len(amounts)):
        for j in range(i + 1, len(amounts)):
            if abs(amounts[i] - amounts[j]) <= 0.05 * abs(amounts[i]):
                similar_count += 1

    similar_ratio = similar_count / len(df) if len(df) > 0 else 0
    return similar_count, similar_ratio


def compute_amount_features(df: pd.DataFrame, time_period: str = None):
    if time_period:
        df = df.resample(time_period, on='TRN_DT').sum()

    absolute_count, absolute_ratio = check_absolute_amount(df)
    subseq_count, trans_ratio, amt_ratio = check_subsequence(df)
    # similar_count, similar_ratio = check_similar_amount(df)

    return {
        'absolute_count': absolute_count,
        'absolute_ratio': absolute_ratio,
        'subseq_count': subseq_count,
        'trans_ratio': trans_ratio,
        'amt_ratio': amt_ratio,
        # 'similar_count': similar_count,
        # 'similar_ratio': similar_ratio
    }


def compute_stats(df: pd.DataFrame, time_period: str = None):
    # 填充缺失值为0
    df['TRN_AMT'].fillna(0, inplace=True)

    # 按正负划分转入和转出
    df['TRN_IN'] = df['TRN_AMT'].apply(lambda x: x if x > 0 else 0)
    df['TRN_OUT'] = df['TRN_AMT'].apply(lambda x: -x if x < 0 else 0)

    if time_period:
        # 重采样并计算统计特征
        resampled = df.resample(time_period, on='TRN_DT')
        diff = (resampled['TRN_IN'].sum() - resampled['TRN_OUT'].sum()).dropna()
        in_amt = resampled['TRN_IN'].sum().dropna()
        out_amt = resampled['TRN_OUT'].sum().dropna()
    else:
        diff = pd.Series(df['TRN_IN'].sum() - df['TRN_OUT'].sum())
        in_amt = pd.Series(df['TRN_IN'].sum())
        out_amt = pd.Series(df['TRN_OUT'].sum())

    # 返回统计特征
    return {
        'diff_mean': diff.mean(),
        'diff_std': diff.std(),
        'diff_max': diff.max(),
        'diff_min': diff.min(),
        'in_mean': in_amt.mean(),
        'in_std': in_amt.std(),
        'in_max': in_amt.max(),
        'in_min': in_amt.min(),
        'out_mean': out_amt.mean(),
        'out_std': out_amt.std(),
        'out_max': out_amt.max(),
        'out_min': out_amt.min(),
    }


def compute_max_values(df: pd.DataFrame, time_period: str = None):
    # 填充缺失值为0
    df['TRN_AMT'].fillna(0, inplace=True)

    # 按正负划分转入和转出
    df['TRN_IN'] = df['TRN_AMT'].apply(lambda x: x if x > 0 else 0)
    df['TRN_OUT'] = df['TRN_AMT'].apply(lambda x: -x if x < 0 else 0)

    df['TRN_OVERALL'] = df['TRN_IN'] + df['TRN_OUT']

    # 定义一个内部函数，用于计算各种统计量
    def compute_resampled_stats(data, column):
        if time_period:
            if data.shape[0] == 0:
                return {
                    f'max_{column}_trn_count': 0,
                    f'max_{column}_trn_amt': 0,
                    f'max_{column}_trn_partner_count': 0
                }
            else:
                resampled = data.resample(time_period, on='TRN_DT')
                return {
                    f'max_{column}_trn_count': resampled.size().max(),
                    f'max_{column}_trn_amt': resampled[f'TRN_{column.upper()}'].sum().max(),
                    f'max_{column}_trn_partner_count': resampled['CRD_TGT'].nunique().max()
                }
        else:
            return {
                f'max_{column}_trn_count': len(data),
                f'max_{column}_trn_amt': data[f'TRN_{column.upper()}'].sum(),
                f'max_{column}_trn_partner_count': data['CRD_TGT'].nunique()
            }

    # 调用内部函数，计算整体、转入、转出的统计量
    overall_stats = compute_resampled_stats(df, "overall")
    in_stats = compute_resampled_stats(df[df['TRN_IN'] > 0], "in")
    out_stats = compute_resampled_stats(df[df['TRN_OUT'] > 0], "out")

    # 合并所有的统计结果
    return {**overall_stats, **in_stats, **out_stats}


def worker(args):
    name, group, time_period = args
    stats = compute_stats(group, time_period)
    max_values = compute_max_values(group, time_period)
    amount_features = compute_amount_features(group, time_period)

    # 将time_period作为前缀，与两个函数返回的字典的键结合，形成新的键
    combined_result = {
        **{f"{time_period}_{key}": value for key, value in stats.items()},
        **{f"{time_period}_{key}": value for key, value in max_values.items()},
        **{f"{time_period}_{key}": value for key, value in amount_features.items()}

    }
    return name, combined_result


def parallel_process(df, periods):
    results = []
    grouped = [(name, group) for name, group in df.groupby('CRD_SRC')]

    # 为每一个time_period生成数据
    for period in periods:
        with Pool(cpu_count()) as pool:
            results.extend(pool.map(worker, [(name, group, period) for name, group in grouped]))

    # 整理数据结构为字典，方便之后创建DataFrame
    data_dict = {}
    for name, stats in results:
        if name not in data_dict:
            data_dict[name] = {}
        data_dict[name].update(stats)

    # 转换为DataFrame
    df_feature = pd.DataFrame(data_dict).T
    df_feature.reset_index(inplace=True)
    df_feature.rename(columns={"index": "CRD_SRC"}, inplace=True)

    return df_feature


def to_time_series():
    aps = to_aps()
    df = parallel_process(aps, time_periods)
    exporter.export_df_to_preprocess('time_series_part1', df)


if __name__ == '__main__':
    start = time.time()
    to_time_series()
    end = time.time()
    print(f'run time all : {(end - start) // 60} minutes.')
