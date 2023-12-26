import os.path
from multiprocessing import cpu_count, Pool, Manager, Lock, Value

import pandas as pd

from constant import dir_preprocess
from data import loader, exporter

# 定义需要处理的时间段
# 如果继续细分，将生成大于样本数量的特征
keys = ['MBANK_TRNFLW_QZ', 'EBANK_CSTLOG_QZ', 'APS_QZ']
time_periods = ['1T']
# time_periods = ['10s', '30s', '1T', '3T', '10T', '1H', '6H', '12H', '1D', '3D', '7D', '14D', '30D']

# 定义全局的 Lock 和 Value 对象
counter_lock = Lock()
counter = Value('i', 0)


def to_refactored_df(key: str):
    column_map = {
        'MBANK_TRNFLW_QZ': ['TFT_CSTNO', 'TFT_DTE_TIME', 'TFT_STDBSNCOD'],
        'EBANK_CSTLOG_QZ': ['CSTNO', 'ADDFIELDDATE', 'BSNCODE'],
        # 'APS_QZ': ['APSDPRDNO', 'APSDTRDAT_TM', 'APSDTRCOD', 'APSDTRAMT'],
        'MBANK_QRYTRNFLW_QZ': ['TFT_CSTNO', 'TFT_DTE_TIME', 'TFT_STDBSNCOD'],
        'EBANK_CSTLOGQUERY_QZ': ['CLQ_CSTNO', 'CLQ_DTE_TIME', 'CLQ_BSNCOD']
    }
    df = loader.to_concat_df(key)
    df = df[column_map[key]]
    df.columns = ['CRD_SRC', 'TRN_DT', 'TRN_COD']
    return df




def worker(name, group, time_period):
    stats = compute_stats(group, time_period)
    max_values = compute_max_values(group, time_period)
    amount_features = compute_amount_features(group, time_period)

    # 将time_period作为前缀，与两个函数返回的字典的键结合，形成新的键
    combined_result = {
        **{f"{time_period}_{key}": value for key, value in stats.items()},
        **{f"{time_period}_{key}": value for key, value in max_values.items()},
        **{f"{time_period}_{key}": value for key, value in amount_features.items()}

    }

    # 更新进度计数器
    with counter_lock:
        counter.value += 1
        # 检查进度
        if counter.value % 1000 == 0:
            print(f"Processed {counter.value} ")

    return name, combined_result


def parallel_process(df, periods):
    results = []
    grouped = [(name, group) for name, group in df.groupby('CRD_SRC')]

    # 为每一个time_period生成数据
    for period in periods:
        print(f'process period -> {period}')
        with Pool(cpu_count()) as pool:
            results.extend(pool.starmap(worker, [(name, group, period) for name, group in grouped]))

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


def to_set_features_part2():
    for key in keys:
        print(f'process key -> {key}')
        df = to_refactored_df(key)
        part2 = parallel_process(df)
        exporter.export_df_to_preprocess(f'{key}_set_part2', part2)


if __name__ == '__main__':
    to_set_features_part2()
