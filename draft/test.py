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


test = getattr(metrics, 'CatKSEvalMetric')
print(test())
