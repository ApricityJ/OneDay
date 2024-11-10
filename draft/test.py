from typing import List
import os
import pickle

from pathlib import Path
import numpy as np
from sklearn.utils import Bunch
import pandas as pd

from data import loader
from util import jsons
from constant import *
import util.metrics as metrics


df = loader.to_df(Path(dir_preprocess).joinpath(f'v7.csv'))
v7_list = df.columns.tolist()
df = loader.to_df(Path(dir_preprocess).joinpath(f'v8.csv'))
v8_list = df.columns.tolist()
print(set(v7_list) - set(v8_list))

