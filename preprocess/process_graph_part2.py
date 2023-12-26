import os
from collections import Counter
from multiprocessing import Pool

import networkx as nx
import pandas as pd

from constant import dir_preprocess
from data import loader, exporter


def node_attribute_features_within_steps(G, node, steps):
    nodes_within_steps = {node}
    frontier = {node}

    for _ in range(steps):
        next_frontier = set()
        for current_node in frontier:
            next_frontier |= set(G.neighbors(current_node))
        frontier = next_frontier - nodes_within_steps
        nodes_within_steps |= next_frontier

    # 计算属性特征
    flag_count = sum(G.nodes[n]['FLAG'] for n in nodes_within_steps)
    total_nodes = len(nodes_within_steps)
    age_list = [G.nodes[n]['AGE'] for n in nodes_within_steps]
    rank_list = [G.nodes[n]['RANK'] for n in nodes_within_steps]
    male_count = sum(1 for n in nodes_within_steps if G.nodes[n]['GENDER'] == 'A')
    female_count = sum(1 for n in nodes_within_steps if G.nodes[n]['GENDER'] == 'B')

    features = {
        'fraud_ratio': flag_count / total_nodes if total_nodes > 0 else 0,
        'avg_age': sum(age_list) / total_nodes if total_nodes > 0 else 0,
        'min_age': min(age_list) if len(age_list) > 0 else 0,
        'max_age': max(age_list) if len(age_list) > 0 else 0,
        'male_ratio': male_count / total_nodes if total_nodes > 0 else 0,
        'female_ratio': female_count / total_nodes if total_nodes > 0 else 0,
        'avg_rank': sum(rank_list) / total_nodes if total_nodes > 0 else 0,
        'min_rank': min(rank_list),
        'max_rank': max(rank_list)
    }

    return features


def edge_attribute_features_within_steps(G, node, steps):
    nodes_within_steps = {node}
    frontier = {node}

    for _ in range(steps):
        next_frontier = set()
        for current_node in frontier:
            next_frontier |= set(G.neighbors(current_node))
        frontier = next_frontier - nodes_within_steps
        nodes_within_steps |= next_frontier

    # 从步长内的节点获取相关的边属性
    trn_amt_list = []
    trn_dt_list = []
    trn_cod_list = []

    for n in nodes_within_steps:
        for neighbor in G[n]:
            edge_data = G[n][neighbor]
            trn_amt_list.append(edge_data['TRN_AMT'])
            trn_dt_list.append(edge_data['TRN_DT'])
            trn_cod_list.append(edge_data['TRN_COD'])

    trn_cod_count = Counter(trn_cod_list)
    most_common_trn_cod, most_common_count = trn_cod_count.most_common(1)[0]

    features = {
        'avg_trn_amt': sum(trn_amt_list) / len(trn_amt_list) if trn_amt_list else 0,
        'max_trn_amt': max(trn_amt_list) if trn_amt_list else 0,
        'min_trn_amt': min(trn_amt_list) if trn_amt_list else 0,
        'total_trn_amt': sum(trn_amt_list) if trn_amt_list > 0 else 0,
        'earliest_trn_dt': min(trn_dt_list) if trn_dt_list else None,
        'latest_trn_dt': max(trn_dt_list) if trn_dt_list else None,
        'most_common_trn_cod': most_common_trn_cod,
        'num_unique_trn_cod': len(set(trn_cod_list))
    }

    return features


def worker(args):
    G, node, step = args

    node_features = node_attribute_features_within_steps(G, node, step)
    edge_features = edge_attribute_features_within_steps(G, node, step)

    return (node, node_features, edge_features)


def parallel_process(G, node_list, steps):
    results = []
    cpu_count = os.cpu_count()

    # 使用多进程进行计算
    for step in steps:
        with Pool(cpu_count) as pool:
            results.extend(pool.map(worker, [(G.copy(), node, step) for node in node_list]))

    # 结果转换为 dataframe
    data = []
    for node, node_features, edge_features in results:
        combined_features = {'CRD_SRC': node}
        combined_features.update(node_features)
        combined_features.update(edge_features)
        data.append(combined_features)

    df = pd.DataFrame(data)

    return df


def to_g_stat():
    steps = ['1', '2']
    G = loader.to_graph('G')
    node_list = loader.to_df(os.path.join(dir_preprocess, 'APS.csv'))['CRD_SRC'].unique().tolist()
    df_g_stst = parallel_process(G, node_list, steps)
    exporter.export_df_to_preprocess('df_g_stst', df_g_stst)


if __name__ == '__main__':
    to_g_stat()
