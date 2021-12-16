# Data Manipulation
import numpy as np
import pandas as pd
import pickle
pd.options.mode.chained_assignment = None 

# System Basics
import os
import math
import datetime
import subprocess
from tqdm import tqdm
from datetime import timedelta
from collections import Counter

# Visualiztion
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt # raincloudplot
from plotnine import  *
import missingno as msno # missing value
from IPython.display import display_html

sns.set_context('paper', font_scale=2)
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize': (14, 8)})

# Preprocessing
## Column Transformation
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.base import TransformerMixin

## Missing value
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
import lightgbm as lgb

## Modelling tools
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold

## imbalance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 

## Calibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

## Evaluating
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay


def mem_size(size_bytes):
    '''
    Convert a file size from B to proper format
    '''
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def reduce_mem_usage(df):
    '''
    Reduce memory cost of a dataframe
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = sum(df.memory_usage())
    print('{:-^55}'.format('Begin downsizing'))
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                    convert_to = "int8"
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16) 
                    convert_to = "int16"
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    convert_to = "int32"
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
                    convert_to = "int64"
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                    convert_to = "float16"
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    convert_to = "float32"
                else:
                    df[col] = df[col].astype(np.float64)
                    convert_to = "float64"
            print(col, "converted from", col_type, "to", convert_to)
    end_mem = sum(df.memory_usage())
    print('{:-^55}'.format('Result'))
    print(f' -> Mem. usage decreased from {mem_size(start_mem)} to {mem_size(end_mem)}')
    print('{:-^55}'.format('Finish downsizing'))
    return df

def summary_memory(df):
    '''
    Calculate the memory cost of each column of a dataframe
    '''
    res = pd.concat([pd.DataFrame(df.memory_usage()).iloc[1:,:].rename({0: 'Memory'}, axis='columns'),
                     pd.DataFrame(df.dtypes).rename({0: 'Data Type'}, axis='columns')],
                    axis=1).reset_index().rename({"index": 'Veriable'}, axis='columns')
    return res


sns.set(rc={'figure.figsize':(12, 6)});
sns.set_style('darkgrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.5f}'.format)

INPUT_PATH = '../input/'

# Features that we finally found important
numc_features = [#'day_of_week',
                 #'day',
                 'hour',
                 #'minute',
                 'click_freq_by_ip',
                 'download_rate_by_ip',
                 'click_freq_by_app',
                 'download_rate_by_app',
                 'click_freq_by_device',
                 'download_rate_by_device',
                 'click_freq_by_os',
                 'download_rate_by_os',
                 'click_freq_by_channel',
                 'download_rate_by_channel',
                 'click_freq_by_hour',
                 'download_rate_by_hour',
                 'click_freq_by_app_channel',
                 'download_rate_by_app_channel',
                 'click_freq_by_ip_hour',
                 'download_rate_by_ip_hour',
                 'click_freq_by_ip_device_app',
                 'download_rate_by_ip_device_app',
                 'click_freq_by_ip_device_os',
                 'download_rate_by_ip_device_os',
                 'click_freq_by_ip_device_os_hour',
                 'download_rate_by_ip_device_os_hour',
                 'nunique_channel_per_ip',
                 'nunique_device_per_ip',
                 'nunique_app_per_ip',
                 'nunique_os_per_ip',
                 'nunique_channel_per_app']
te_features = []
ohe_features = []