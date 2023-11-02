import os
from multiprocessing import cpu_count, Pool

import pandas as pd

from constant import dir_preprocess
from data import loader, exporter


def to_aps():
    df = loader.to_df(os.path.join(dir_preprocess, 'APS.csv'))
    df['TRN_DT'] = pd.to_datetime(df['TRN_DT'], format='%Y%m%d%H%M%S')
    return df


def night_day_stats(df):
    # 定义夜间时间段
    night_hours = set(range(0, 6))  # 凌晨0点至6点为夜间

    # 提取交易时间中的小时
    df['hour'] = df['TRN_DT'].dt.hour

    # 计算夜间和白天的交易次数和交易总金额
    night_trans_count = df[df['hour'].isin(night_hours)].shape[0]
    night_trans_amount = df[df['hour'].isin(night_hours)]['TRN_AMT'].abs().sum()

    day_trans_count = df[~df['hour'].isin(night_hours)].shape[0]
    day_trans_amount = df[~df['hour'].isin(night_hours)]['TRN_AMT'].abs().sum()

    total_count = df.shape[0]
    total_amount = df['TRN_AMT'].abs().sum()

    # 计算占比
    night_count_ratio = night_trans_count / total_count if total_count > 0 else 0
    night_amount_ratio = night_trans_amount / total_amount if total_amount > 0 else 0
    day_count_ratio = day_trans_count / total_count if total_count > 0 else 0
    day_amount_ratio = day_trans_amount / total_amount if total_amount > 0 else 0

    night_day_count_ratio = night_trans_count / day_trans_count if day_trans_count > 0 else 0
    night_day_amount_ratio = night_trans_amount / day_trans_amount if day_trans_amount > 0 else 0

    return {
        'night_trans_count': night_trans_count,
        'night_trans_amount': night_trans_amount,
        'day_trans_count': day_trans_count,
        'day_trans_amount': day_trans_amount,
        'night_count_ratio': night_count_ratio,
        'night_amount_ratio': night_amount_ratio,
        'day_count_ratio': day_count_ratio,
        'day_amount_ratio': day_amount_ratio,
        'night_day_count_ratio': night_day_count_ratio,
        'night_day_amount_ratio': night_day_amount_ratio
    }


def weekend_weekday_stats(df):
    # 根据交易日期提取星期信息 (Monday=0, Sunday=6)
    df['weekday'] = df['TRN_DT'].dt.weekday

    # 计算周末和工作日的交易次数和交易总金额
    weekend_trans_count = df[df['weekday'] >= 5].shape[0]
    weekend_trans_amount = df[df['weekday'] >= 5]['TRN_AMT'].abs().sum()

    weekday_trans_count = df[df['weekday'] < 5].shape[0]
    weekday_trans_amount = df[df['weekday'] < 5]['TRN_AMT'].abs().sum()

    total_count = df.shape[0]
    total_amount = df['TRN_AMT'].abs().sum()

    # 计算占比
    weekend_count_ratio = weekend_trans_count / total_count if total_count > 0 else 0
    weekend_amount_ratio = weekend_trans_amount / total_amount if total_amount > 0 else 0
    weekday_count_ratio = weekday_trans_count / total_count if total_count > 0 else 0
    weekday_amount_ratio = weekday_trans_amount / total_amount if total_amount > 0 else 0

    weekend_weekday_count_ratio = weekend_trans_count / weekday_trans_count if weekday_trans_count > 0 else 0
    weekend_weekday_amount_ratio = weekend_trans_amount / weekday_trans_amount if weekday_trans_amount > 0 else 0

    return {
        'weekend_trans_count': weekend_trans_count,
        'weekend_trans_amount': weekend_trans_amount,
        'weekday_trans_count': weekday_trans_count,
        'weekday_trans_amount': weekday_trans_amount,
        'weekend_count_ratio': weekend_count_ratio,
        'weekend_amount_ratio': weekend_amount_ratio,
        'weekday_count_ratio': weekday_count_ratio,
        'weekday_amount_ratio': weekday_amount_ratio,
        'weekend_weekday_count_ratio': weekend_weekday_count_ratio,
        'weekend_weekday_amount_ratio': weekend_weekday_amount_ratio
    }


def worker(args):
    name, group = args
    night_day = night_day_stats(group)
    weekend_weekday = weekend_weekday_stats(group)

    combined_result = {
        **night_day,
        **weekend_weekday
    }

    return name, combined_result


def parallel_process(df):
    results = []
    grouped = [(name, group) for name, group in df.groupby('CRD_SRC')]

    with Pool(cpu_count()) as pool:
        results.extend(pool.map(worker, [(name, group) for name, group in grouped]))

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
    df = parallel_process(aps)
    exporter.export_df_to_preprocess('time_series_part2', df)


if __name__ == '__main__':
    to_time_series()
