import os

import networkx as nx
import pandas as pd

from constant import dir_preprocess
from data import loader, exporter


# 放平心态 - 精细化留待以后
def to_aps():
    df = loader.to_df(os.path.join(dir_preprocess, 'APS.csv'))
    df['TRN_DT'] = pd.to_datetime(df['TRN_DT'], format='%Y%m%d%H%M%S')
    return df


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
        df = to_aps()

        # 批量添加节点
        payment_nodes = df['CRD_SRC'].dropna().tolist()
        receiving_nodes = df['CRD_TGT'].dropna().tolist()
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

        df.dropna(subset=['CRD_TGT'], inplace=True)
        # 批量添加边，并将时间和金额添加到边属性
        for _, row in df.iterrows():
            G.add_edge(row['CRD_SRC'], row['CRD_TGT'],
                       time=row['TRN_DT'],
                       amount=row['TRN_AMT'],
                       code=row['TRN_COD'])

        exporter.export_g_to_preprocess('G', G)

    return G


# 计算节点'平庸的'特征，生成dataframe
def to_g_ordinary():
    # 指定的图
    G = to_graph('G')

    # 指定的数据源
    df = to_aps()
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

    exporter.export_df_to_preprocess('g_ordinary', df_merged)


if __name__ == '__main__':
    to_graph('G')
    # to_g_ordinary()
