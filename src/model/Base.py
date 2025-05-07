import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OUTDATED_IGNORE'] = '1'

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 100

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

import numpy as np
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import confusion_matrix, r2_score  
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier                  

from abc import ABC
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from joblib import dump, load
from time import time
from functools import wraps
from IPython.display import display, clear_output



def validate_split_size(func):

    @wraps(func)
    def wrapper(self, data, **kwargs):
        split_size = kwargs.get('split_size', (0.7, 0.15, 0.15))

        if not isinstance(split_size, tuple):
            print("split_size deve ser uma tupla!")

        elif len(split_size) != 3:
            print("split_size deve ter 3 elementos")

        elif any(ss <= 0 for ss in split_size):
            print("Todos os valores em 'split_size' devem ser maiores que zero.")
        
        elif sum(split_size) != 1:
            print("A soma dos valores em 'split_size' deve ser igual a 1.")
        
        else:
            return func(self, data, **kwargs)
        
    return wrapper


def validate_task(func):

    @wraps(func)
    def wrapper(self, data, **kwargs):
        task = kwargs.get('task', None)

        if task is not None and task not in ['regression', 'classification']:
            print("Já que deseja informar a task explicitamente, ela deve ser 'regression' ou 'classification'")
        
        else:
            return func(self, data, **kwargs)
        
    return wrapper


def validate_target(func):

    @wraps(func)
    def wrapper(self, data, **kwargs):

        if self.target not in data.columns:
            print(f"O dataset não contém a variável '{self.target}'")
        
        else:
            return func(self, data, **kwargs)
        
    return wrapper


class Base(ABC):

    def __init__(self, target, name, seed=None):

        self.target = target
        self.name = name
        self.seed = seed
        self.preprocessor = None
        self.preprocessed_data = None
        self.task  = None
        self.model = None
        self.hyperparameter = None
        self.history_kfold = None
        self.have_cat = False


    def build(self):
        pass

    def _optimizer(self):
        pass

    def hyperparameter_optimization(self):
        pass

    def load(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass
    
    def _preprocess(self, data, target_one_hot_encoder=False, **kwargs):

        max_cat_nunique = kwargs.get('max_cat_nunique', 10)
        split_size = kwargs.get('split_size', (0.7, 0.15, 0.15))
        info = kwargs.get('info', True)
        task = kwargs.get('task', None)  
        despise = kwargs.get('despise', [])  
        shuffle_split = kwargs.get('shuffle_split', True)  
        drop_intersection_time_series = kwargs.get('drop_intersection_time_series', None) 

        train_size = split_size[0]
        valid_size = split_size[1]
        test_size = split_size[2]

        _data = data.copy()
        _data.drop(despise, axis=1, inplace=True)

        num_rows_with_nan = _data.isna().any(axis=1).sum()
        _data.dropna(axis=0, inplace=True)
        
        for feature in _data.columns:
            if _data[feature].dtype == 'object' and pd.to_numeric(_data[feature], errors='coerce').notna().all():
                _data[feature] = pd.to_numeric(_data[feature])

        types_num = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                     'float16', 'float32', 'float64']

    
        ######### X #########
        _X = _data.drop(columns=[self.target])
        df_preprocessor = None

        categorical_cols = list()
        numerical_cols= list()
        high_cardinality_cols = list()

        for feature in _X.columns:

            if _X[feature].dtype in types_num:
                numerical_cols.append(feature)

            elif np.unique(_X[feature]).size <= max_cat_nunique:
                categorical_cols.append(feature)
                self.have_cat = True

            else:
                high_cardinality_cols.append(feature)

        
        _X[categorical_cols] = _X[categorical_cols].astype('category')

        df_preprocessor_num = Pipeline(steps=[
            ('standardlization_num', StandardScaler(with_mean=True))
        ])
    
        df_preprocessor_cat = Pipeline(steps=[
            ('onehot_cat', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        df_preprocessor = ColumnTransformer(
            transformers=[
                ('df_preprocessor_num', df_preprocessor_num, numerical_cols),
                ('df_preprocessor_cat', df_preprocessor_cat, categorical_cols)
            ],
            remainder='drop',
            sparse_threshold=0
        )
    
        ######### Y #########
        _y = _data[[self.target]]
        target_preprocessor = None

        if task in ['classification', 'regression']:
            self.task = task

        elif _y[self.target].dtypes not in types_num:
            self.task = 'classification'

        elif np.unique(_y[self.target]).size > max_cat_nunique:
            self.task = 'regression'

        else:
            self.task = 'classification'


        ######### transformation #########
        if self.task == 'classification':

            if target_one_hot_encoder:
                encoder = OneHotEncoder(handle_unknown='ignore')

            else:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
            
            target_preprocessor_cat = Pipeline(steps=[
                ('target_encoder_cat', encoder)
            ])
            
            target_preprocessor = ColumnTransformer(
                transformers=[
                    ('target_preprocessor_cat', target_preprocessor_cat, [self.target])
                ],
                remainder='drop',
                sparse_threshold=0
            )
    
            _y = target_preprocessor.fit_transform(_y)
        
        else:
            # _y = _y.to_numpy()

            target_preprocessor_num = Pipeline(steps=[
                ('standardlization_num', StandardScaler(with_mean=True))
            ])
           
            target_preprocessor = ColumnTransformer(
                transformers=[
                    ('target_preprocessor_cat', target_preprocessor_num, [self.target])
                ],
                remainder='drop',
                sparse_threshold=0
            )
    
            _y = target_preprocessor.fit_transform(_y)
    
        ######### TRAIN / VALID / TEST SPLIT #########
        _X_train, _X_temp, y_train, _y_temp = train_test_split(_X, _y, test_size=1-train_size, shuffle=shuffle_split)

        X_train = df_preprocessor.fit_transform(_X_train)
        _X_temp = df_preprocessor.transform(_X_temp)

        # Eliminação de intersecção dos dados
        if drop_intersection_time_series is not None:
            X_train = X_train[:-drop_intersection_time_series, :]
            _X_temp = _X_temp[drop_intersection_time_series:, :]

            y_train = y_train[:-drop_intersection_time_series, :]
            _y_temp = _y_temp[drop_intersection_time_series:, :]
    
        X_val, X_test, y_val, y_test = train_test_split(_X_temp, _y_temp, test_size=test_size/(valid_size + test_size), shuffle=shuffle_split)
        
        # Eliminação de intersecção dos dados
        if drop_intersection_time_series is not None:
            X_val = X_val[:-drop_intersection_time_series, :]
            X_test = X_test[drop_intersection_time_series:, :]

            y_val = y_val[:-drop_intersection_time_series, :]
            y_test = y_test[drop_intersection_time_series:, :]
    
        X_train_val = np.concatenate((X_train, X_val), axis=0)
        y_train_val = np.concatenate((y_train, y_val), axis=0)
        
        if not target_one_hot_encoder:
            y_train = y_train.reshape(-1)
            y_val = y_val.reshape(-1)
            y_train_val = y_train_val.reshape(-1)
            y_test = y_test.reshape(-1)
        
        self.preprocessed_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_train_val': X_train_val,
            'y_train_val': y_train_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
        self.preprocessor = {
            "features": df_preprocessor,
            "target": target_preprocessor 
        }

        if info:
            msg = f"""
                Task: {self.task}

                Total of registers: {len(data)}
                Total of valid registers: {len(_X)}
                Total of invalid registers: {num_rows_with_nan}

                Total of training registers: {X_train.shape[0]}
                Total of validation registers: {X_val.shape[0]}
                Total of test registers: {X_test.shape[0]}

                Features before preprocessing: {_X_train.shape[1]}
                Features after preprocessing: {X_train.shape[1]}

                Numerical Features: {numerical_cols}
                Categorical Features: {categorical_cols}
                Categorical Features removed due to high cardinality: {high_cardinality_cols}

                Target: ['{self.target}']
            """

            if self.task == 'classification':
                if target_one_hot_encoder == False:
                    msg += f"\tCardinality (Target): {np.unique(self.preprocessed_data['y_train']).size}"

                else:
                    msg += f"\tCardinality (Target): {self.preprocessed_data['y_train'].shape[1]}"

            print(msg)