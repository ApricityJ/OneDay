import os
from multiprocessing import Queue, cpu_count, Process

import networkx as nx
import pandas as pd

from constant import dir_preprocess
from data import loader, exporter


def to_neighbors_and_edges_within_steps(G, node, steps):
    lengths = nx.single_source_shortest_path_length(G, node, cutoff=steps)
    neighbors = [k for k, v in lengths.items() if v <= steps and k != node]

    neighbors_data = []

    for neighbor in neighbors:
        attr = G.nodes[neighbor]
        row = {
            'step': lengths[neighbor],
            'node': neighbor,
            'FLAG': attr.get('flag'),
            'AGE': attr.get('age'),
            'GENDER': attr.get('gender'),
            'RANK': attr.get('rank')
        }
        neighbors_data.append(row)

    # 获取与这些邻居相关的边及其属性
    edges_data = []
    for neighbor in neighbors:
        if G.has_edge(node, neighbor):
            edge_attr = G[node][neighbor]
            edge_data = {
                'src_node': node,
                'tgt_node': neighbor,
                'TRN_DT': edge_attr.get('trn_dt'),
                'TRN_AMT': edge_attr.get('trn_amt'),
                'TRN_COD': edge_attr.get('trn_cod')
            }
            edges_data.append(edge_data)

    neighbors_df = pd.DataFrame(neighbors_data)
    edges_df = pd.DataFrame(edges_data)

    return neighbors_df, edges_df


def producer(G, node_list, steps, task_queue):
    for node in node_list:
        neighbors_df, edges_df = to_neighbors_and_edges_within_steps(G, node, steps)
        task = {'node': node, 'step': steps, 'neighbors_data': neighbors_df, 'edges_data': edges_df}
        task_queue.put(task)
    for _ in range(cpu_count() - 2):  # 结束信号，告诉消费者进程结束
        task_queue.put(None)


def to_edge_features(edges_df):
    """
    根据边的DataFrame提取特征。

    参数:
    - edges_df: 与节点相关的边及其属性的DataFrame

    返回:
    - 特征字典
    """
    if edges_df.empty:
        return {
            'total_transaction_amount': 0,
            'average_transaction_amount': 0,
            'max_transaction_amount': 0,
            'min_transaction_amount': 0,
            'total_transactions': 0,
            'unique_trn_codes_ratio': 0,
            'most_common_trn_code_ratio': 0,
            'unique_trn_days_ratio': 0,
            'transaction_amount_std': 0,
            'transaction_amount_median': 0,
            'transaction_amount_25th_percentile': 0,
            'transaction_amount_75th_percentile': 0
        }

    total_transaction_amount = edges_df['TRN_AMT'].sum()
    average_transaction_amount = edges_df['TRN_AMT'].mean()
    max_transaction_amount = edges_df['TRN_AMT'].max()
    min_transaction_amount = edges_df['TRN_AMT'].min()
    total_transactions = len(edges_df)

    trn_code_counts = edges_df['TRN_COD'].value_counts()
    unique_trn_codes_ratio = trn_code_counts.nunique() / total_transactions if total_transactions > 0 else 0
    most_common_trn_code_ratio = trn_code_counts.iloc[0] / total_transactions if len(
        trn_code_counts) > 0 and total_transactions > 0 else 0

    trn_days = pd.to_datetime(edges_df['TRN_DT'])
    unique_trn_days_ratio = trn_days.nunique() / total_transactions if total_transactions > 0 else 0

    transaction_amount_std = edges_df['TRN_AMT'].std()
    transaction_amount_median = edges_df['TRN_AMT'].median()
    transaction_amount_25th_percentile = edges_df['TRN_AMT'].quantile(0.25)
    transaction_amount_75th_percentile = edges_df['TRN_AMT'].quantile(0.75)

    return {
        'total_transaction_amount': total_transaction_amount,
        'average_transaction_amount': average_transaction_amount,
        'max_transaction_amount': max_transaction_amount,
        'min_transaction_amount': min_transaction_amount,
        'total_transactions': total_transactions,
        'unique_trn_codes_ratio': unique_trn_codes_ratio,
        'most_common_trn_code_ratio': most_common_trn_code_ratio,
        'unique_trn_days_ratio': unique_trn_days_ratio,
        'transaction_amount_std': transaction_amount_std,
        'transaction_amount_median': transaction_amount_median,
        'transaction_amount_25th_percentile': transaction_amount_25th_percentile,
        'transaction_amount_75th_percentile': transaction_amount_75th_percentile
    }


def to_node_features(neighbors_df):
    """
    根据节点的DataFrame提取特征。

    参数:
    - neighbors_df: 邻居及其属性的DataFrame

    返回:
    - 特征字典
    """
    if neighbors_df.empty:
        return {
            'num_neighbors': 0,
            'num_flagged_neighbors': 0,
            'average_age': 0,
            'gender_A_ratio': 0,  # A表示男性
            'gender_B_ratio': 0,  # B表示女性
            **{'rank_{}_count'.format(i): 0 for i in range(1, 7)}  # 对每个等级的统计
        }

    num_neighbors = len(neighbors_df)
    num_flagged_neighbors = neighbors_df['FLAG'].sum()
    average_age = neighbors_df['AGE'].mean()

    gender_counts = neighbors_df['GENDER'].value_counts(normalize=True)
    gender_A_ratio = gender_counts.get('A', 0)
    gender_B_ratio = gender_counts.get('B', 0)

    rank_counts = neighbors_df['RANK'].value_counts(normalize=False)
    rank_stats = {'rank_{}_count'.format(rank): rank_counts.get(rank, 0) for rank in range(1, 7)}

    return {
        'num_neighbors': num_neighbors,
        'num_flagged_neighbors': num_flagged_neighbors,
        'average_age': average_age,
        'gender_A_ratio': gender_A_ratio,
        'gender_B_ratio': gender_B_ratio,
        **rank_stats
    }


def consumer(task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            result_queue.put(task)
            break

        node = task['node']
        neighbors_df = task['neighbors_data']
        edges_df = task['edges_data']

        node_features = to_node_features(neighbors_df)
        edge_features = to_edge_features(edges_df)

        # 合并两个特征字典
        features = {**node_features, **edge_features, 'node': node}

        result_queue.put(features)


def collector(result_queue):
    data = []
    completed_processes = 0
    while completed_processes < (cpu_count() - 2):  # 8个消费者进程
        result = result_queue.get()
        if result is None:  # 一个消费者进程已经完成
            completed_processes += 1
        else:
            data.append(result)
    df = pd.DataFrame(data)
    exporter.export_df_to_preprocess('g_stat', df)


if __name__ == "__main__":
    G = loader.to_graph('G')
    node_list = loader.to_df(os.path.join(dir_preprocess, 'APS.csv'))['CRD_SRC'].unique().tolist()
    node_list = node_list
    task_queue = Queue()
    result_queue = Queue()

    steps = 2
    p = Process(target=producer, args=(G, node_list, steps, task_queue))
    p.start()

    consumers = []
    for _ in range(cpu_count() - 2):
        c = Process(target=consumer, args=(task_queue, result_queue))
        c.start()
        consumers.append(c)

    collector_process = Process(target=collector, args=(result_queue,))
    collector_process.start()

    p.join()
    for c in consumers:
        c.join()
    collector_process.join()
