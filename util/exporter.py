import networkx as nx
from networkx import DiGraph
from pandas import DataFrame
import pickle

from constant import *


def to_graphml(p: str, n: str, g: DiGraph):
    with open(os.path.join(p, f'{n}.pkl'), 'wb') as f:
        pickle.dump(g, f)


def export_g_to_preprocess(n: str, g: DiGraph):
    to_graphml(dir_preprocess, n, g)


def to_csv(p: str, n: str, df: DataFrame):
    df.to_csv(os.path.join(p, f'{n}.csv'), index=False)


def export_ds(p: str, ds: dict):
    for k, v in ds.items():
        to_csv(p, k, v)


def export_df_to_train(n: str, df: DataFrame):
    to_csv(dir_train, n, df)


def export_df_to_preprocess(n: str, df: DataFrame):
    to_csv(dir_preprocess, n, df)


def export_df_to_test(n: str, df: DataFrame):
    to_csv(dir_test, n, df)


def export_ds_to_preprocess(ds: dict):
    export_ds(dir_preprocess, ds)


def export_ds_to_result(ds: dict):
    export_ds(dir_result, ds)


def to_submission(df: DataFrame):
    return df.to_csv(dir_result.joinpath('result.csv'), header=False)
