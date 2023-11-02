import os.path

import networkx as nx
import pandas as pd
from pandas import DataFrame
from networkx import DiGraph

from data import loader, exporter

# 定义一组常量
graph_keys = ['MBANK_TRNFLW_QZ', 'EBANK_CSTLOG_QZ', 'APS_QZ']

# 包含关联关系的相关表
time_columns = {
    'MBANK_TRNFLW_QZ': {'TFT_DTE_TIME': '%Y%m%d%H%M%S'},
    'EBANK_CSTLOG_QZ': {'ADDFIELDDATE': '%Y%m%d'},
    'APS_QZ': {'APSDTRDAT_TM': '%Y%m%d%H%M%S'}
}

# 关联关系的字段，同时进行了排序
relation_columns = {
    'MBANK_TRNFLW_QZ': ['TFT_CSTACC', 'TFT_DTE_TIME', 'TFT_STDBSNCOD', 'TFT_TRNAMT', 'TFT_CRPACC'],
    'EBANK_CSTLOG_QZ': ['FRMACCTNO', 'ADDFIELDDATE', 'BSNCODE', 'TRNAMT', 'TOACCTNO'],
    'APS_QZ': ['APSDPRDNO', 'APSDTRDAT_TM', 'APSDTRCOD', 'APSDTRAMT', 'APSDCPTPRDNO'],
}


# 根据key获取一个dataframe
def to_df_by_key(k: str):
    df = loader.to_concat_df(k)[relation_columns[k]]
    d = time_columns[k]
    for col, fmt in d.items():
        df[col] = pd.to_datetime(df[col], format=fmt)
    # 重命名
    # 付款卡，时间，交易码，金额，收款卡
    df.columns = ['CRD_SRC', 'TRN_DT', 'TRN_COD', 'TRN_AMT', 'CRD_TGT']
    return df


# 读取这些表生成一个dataset
# key : dataframe 包含train和test，使用src区分
def to_ds_time_series():
    ds = {}
    for k, v in time_columns.items():
        ds[k] = to_df_by_key(k)
    return ds


# 生成图
def to_graph(key: str):
    G = loader.to_graph(key)
    # 不存在则初始化
    if not G:

        G = nx.DiGraph()

        # 为节点标注属性，目前标注TARGET:FLAG,NATURE_CUST:其它
        # 事实上有太多可以标注在这里的，例如交易行为的特征
        # 先退出深度优先，如有时间再优化

        target = loader.to_concat_df('TARGET_QZ')
        cust = loader.to_concat_df('NATURE_CUST_QZ')
        df_cust_info = target.merge(cust, left_on=['CUST_NO', 'SRC'], right_on=['CUST_NO', 'SRC'], how='left')

        # 图原始数据
        df = to_df_by_key(key)

        # 批量添加节点
        payment_nodes = df['CRD_SRC'].tolist()
        receiving_nodes = df['CRD_TGT'].tolist()
        all_nodes = set(payment_nodes + receiving_nodes)

        # 筛选这些节点的节点特征
        df_node = pd.DataFrame({'CARD_NO': list(all_nodes)})
        df_node_info = df_node.merge(df_cust_info, left_on=['CARD_NO'], right_on=['CARD_NO'], how='left')
        df_node_info = df_node_info[['CARD_NO', 'FLAG', 'NTRL_CUST_AGE', 'NTRL_CUST_SEX_CD', 'NTRL_RANK_CD']]
        df_node_info.columns = ['CARD_NO', 'FLAG', 'AGE', 'GENDER', 'RANK']

        for _, row in df_node_info.iterrows():
            # 添加节点，并附带属性
            G.add_node(row['CARD_NO'],
                       flag=row['FLAG'],
                       age=row['AGE'],
                       gender=row['GENDER'],
                       rank=row['RANK'])

        # 批量添加边，并将时间和金额添加到边属性
        for _, row in df.iterrows():
            G.add_edge(row['CRD_SRC'], row['CRD_TGT'],
                       time=row['TRN_DT'],
                       amount=row['TRN_AMT'],
                       code=row['TRN_COD'])

        exporter.export_g_to_preprocess(key, G)

    return G


# 计算节点'平庸的'特征，生成dataframe
def to_df_feature_ordinary(key: str):
    # 指定的图
    G = to_graph(key)

    # 指定的数据源
    df = to_df_by_key(key)
    df_node_src = df['CRD_SRC'].unique()

    # 计算节点度
    degree_dict = dict(G.degree())

    # 计算节点的度中心性
    degree_centrality = nx.degree_centrality(G)
    degree_closeness_centrality = nx.closeness_centrality(G)
    degree_betweenness_centrality = nx.betweenness_centrality(G)

    # 计算节点的聚类系数
    clustering_coefficient = nx.clustering(G)

    # 计算节点的 PageRank 值
    pagerank = nx.pagerank(G)

    # 创建一个包含节点特征的 Pandas DataFrame
    df_node_feature = pd.DataFrame({
        'Node': list(G.nodes()),
        'Degree': list(degree_dict.values()),
        'Degree_Centrality': list(degree_centrality.values()),
        'Degree_Closeness_Centrality': list(degree_closeness_centrality.values()),
        'Degree_Betweenness_Centrality': list(degree_betweenness_centrality.values()),
        'Clustering_Coefficient': list(clustering_coefficient.values()),
        'PageRank': list(pagerank.values())
    })

    # 返回需要的节点的特征
    df_merged = df_node_feature.merge(df_node_src, left_on=['Node'], right_on=['CRD_SRC'], how='inner')
    df_merged.drop(columns=['CRD_SRC'], inplace=True)

    return df_merged


# 计算节点邻居属性的特征
def to_feature_of_nodes_in_N_steps(G: DiGraph, node: str, N: int):
    # 第N步的周边
    nodes_within_N_steps = list(nx.single_source_shortest_path_length(G, node, cutoff=N).keys())

    # 收集这些节点的属性
    data = []
    for node in nodes_within_N_steps:
        data.append({
            'id': node,
            'age': G.nodes[node]['age'],
            'rank': G.nodes[node]['rank'],
            'label': G.nodes[node]['label']
        })

    # 转换为dataframe
    df_raw = pd.DataFrame(data)

    # 转换为dict
    d = {}
    for col in df_raw.columns.tolist():
        d[col] = df_raw.loc[0, col]

    return d


# 处理一张表的N步邻居特征
def to_df_feature_of_nodes_in_N_steps(key: str):
    steps = [1, 2, 3]
    g = to_graph(key)

    df = to_df_by_key(key)
    node_list = df['CRD_SRC'].unique().tolist()

    dict_list = []
    for node in node_list:
        d = {}
        # 依次处理1，2，3步的特征
        for step in steps:
            f = to_feature_of_nodes_in_N_steps(g, node, step)
            d.update(f)
        dict_list.append(d)

    return DataFrame(dict_list)


if __name__ == '__main__':
    # 初始化
    for k in graph_keys:
        G = to_graph(k)
        df_ord = to_df_feature_ordinary(k)
        df_n_step = to_df_feature_of_nodes_in_N_steps(k)
