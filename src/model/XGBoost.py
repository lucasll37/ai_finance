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

from abc import ABC
from xgboost import XGBClassifier, XGBRegressor
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from joblib import dump, load
from time import time
from functools import wraps
from IPython.display import display, clear_output
from .Base import Base


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


class XgBoost(Base):

    def __init__(self, target, name=None, seed=None):

        if name is None:
            _name = f'xgBoost {datetime.now().strftime("%d-%m-%y %Hh%Mmin")}'

        else:
            _name = name

        super().__init__(target, _name, seed)
        
        self._metrics = None
        self.patience_early_stopping = None
          

    @validate_target
    @validate_task
    @validate_split_size
    def build(self, data, **kwargs):

        super()._preprocess(data, **kwargs)
        
        if self.task == 'regression':
            self._metrics = ['rmse']

        else:
            self._metrics = ['mlogloss']


        self.patience_early_stopping = kwargs.get('patience_early_stopping', 20)


    def _make_xgBooster(self, tree_method, booster, learning_rate, min_split_loss, max_depth,
                        min_child_weight, max_delta_step, subsample, sampling_method,
                        colsample_bytree, colsample_bylevel, colsample_bynode, reg_lambda,
                        reg_alpha, scale_pos_weight, grow_policy, max_leaves, max_bin,
                        num_parallel_tree, verbose=0):

        common_arguments = {
            'tree_method': tree_method,
            'n_estimators': 100_000,
            'early_stopping_rounds': self.patience_early_stopping,
            'booster': booster,
            'eval_metric': self._metrics,
            'validate_parameters': False,
            'learning_rate': learning_rate,
            'min_split_loss': min_split_loss,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'max_delta_step': max_delta_step,
            'subsample': subsample,
            'sampling_method': sampling_method,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bynode': colsample_bynode,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'scale_pos_weight': scale_pos_weight,
            'grow_policy': grow_policy,
            'max_leaves': max_leaves,
            'max_bin': max_bin,
            'num_parallel_tree': num_parallel_tree,
            'random_state': self.seed,
            'verbosity': verbose
        } 

        if self.task == "regression":
            model = XGBRegressor(
                objective='reg:squarederror',
                **common_arguments
            )

        else:
            model = XGBClassifier(
                objective='multi:softprob',
                num_class= np.unique(self.preprocessed_data['y_train']).size,
                use_label_encoder=False,
                **common_arguments
            )
            
        return model

    def _optimizer(self, trial, **kwargs):

        num_folds = kwargs.get('num_folds', 5)
        info = kwargs.get('info', False)
        shuffle_kfold = kwargs.get('shuffle_kfold', True)

        search_space_tree_method = kwargs.get('search_space_tree_method', ['auto'])
        search_space_booster = kwargs.get('search_space_booster', ['gbtree', 'gblinear', 'dart'])
        search_space_learning_rate = kwargs.get('search_space_learning_rate', [0.1, 0.3])
        search_space_min_split_loss = kwargs.get('search_space_min_split_loss', [0])
        search_space_max_depth = kwargs.get('search_space_max_depth', [5, 6, 7])
        search_space_min_child_weight = kwargs.get('search_space_min_child_weight', [1])
        search_space_max_delta_step = kwargs.get('search_space_max_delta_step', [0])
        search_space_subsample = kwargs.get('search_space_subsample', [1])
        search_space_sampling_method = kwargs.get('search_space_sampling_method', ['uniform'])
        search_space_colsample_bytree = kwargs.get('search_space_colsample_bytree', [1])
        search_space_colsample_bylevel = kwargs.get('search_space_colsample_bylevel', [1])
        search_space_colsample_bynode = kwargs.get('search_space_colsample_bynode', [1])
        search_space_reg_lambda = kwargs.get('search_space_reg_lambda', [1])
        search_space_reg_alpha = kwargs.get('search_space_reg_alpha', [0])
        search_space_scale_pos_weight = kwargs.get('search_space_scale_pos_weight', [1])
        search_space_grow_policy = kwargs.get('search_space_grow_policy', ['depthwise', 'lossguide'])
        search_space_max_leaves = kwargs.get('search_space_max_leaves', [0])
        search_space_max_bin = kwargs.get('search_space_max_bin', [256])
        search_space_num_parallel_tree = kwargs.get('search_space_num_parallel_tree', [1])

        tree_method = trial.suggest_categorical('tree_method', search_space_tree_method)
        booster = trial.suggest_categorical('booster', search_space_booster)
        learning_rate = trial.suggest_categorical('learning_rate', search_space_learning_rate)
        min_split_loss = trial.suggest_categorical('min_split_loss', search_space_min_split_loss)
        max_depth = trial.suggest_categorical('max_depth', search_space_max_depth)
        min_child_weight = trial.suggest_categorical('min_child_weight', search_space_min_child_weight)
        max_delta_step = trial.suggest_categorical('max_delta_step', search_space_max_delta_step)
        subsample = trial.suggest_categorical('subsample', search_space_subsample)
        sampling_method = trial.suggest_categorical('sampling_method', search_space_sampling_method)
        colsample_bytree = trial.suggest_categorical('colsample_bytree', search_space_colsample_bytree)
        colsample_bylevel = trial.suggest_categorical('colsample_bylevel', search_space_colsample_bylevel)
        colsample_bynode = trial.suggest_categorical('colsample_bynode', search_space_colsample_bynode)
        reg_lambda = trial.suggest_categorical('reg_lambda', search_space_reg_lambda)
        reg_alpha = trial.suggest_categorical('reg_alpha', search_space_reg_alpha)
        scale_pos_weight = trial.suggest_categorical('scale_pos_weight', search_space_scale_pos_weight)
        grow_policy = trial.suggest_categorical('grow_policy', search_space_grow_policy)
        max_leaves = trial.suggest_categorical('max_leaves', search_space_max_leaves)
        max_bin = trial.suggest_categorical('max_bin', search_space_max_bin)
        num_parallel_tree = trial.suggest_categorical('num_parallel_tree', search_space_num_parallel_tree)

        kfold = KFold(n_splits=num_folds, shuffle=shuffle_kfold)

        scores = []

        X_train_val = self.preprocessed_data['X_train_val']
        y_train_val = self.preprocessed_data['y_train_val']

        for index_train, index_val in kfold.split(X_train_val, y_train_val):

            modelStudy = self._make_xgBooster(
                tree_method = tree_method,
                booster = booster,
                learning_rate = learning_rate,
                min_split_loss = min_split_loss,
                max_depth = max_depth,
                min_child_weight = min_child_weight,
                max_delta_step = max_delta_step,
                subsample = subsample,
                sampling_method = sampling_method,
                colsample_bytree = colsample_bytree,
                colsample_bylevel = colsample_bylevel,
                colsample_bynode = colsample_bynode,
                reg_lambda = reg_lambda,
                reg_alpha = reg_alpha,
                scale_pos_weight = scale_pos_weight,
                grow_policy = grow_policy,
                max_leaves = max_leaves,
                max_bin = max_bin,
                num_parallel_tree = num_parallel_tree
            )
        
            modelStudy.fit(
                X_train_val[index_train],
                y_train_val[index_train],
                eval_set = [(
                    X_train_val[index_val],
                    y_train_val[index_val]
                )],
                verbose=0
            )

            scores.append(modelStudy.best_score)

        new_trial = pd.DataFrame([scores], columns=self.history_kfold.columns)
        self.history_kfold = pd.concat([self.history_kfold, new_trial], ignore_index=True)
        self.history_kfold.rename_axis('Trial (nº)', inplace=True)

        if info:
            trial = self.hyperparameter.trials_dataframe()
            trial = trial.set_index("number")
            trial.rename_axis('Trial (nº)', inplace=True)
            trial.rename(columns={'value': 'Folds mean'}, inplace=True)

            temp = self.history_kfold.copy() 
            # temp['Folds std'] = temp.std(axis=1) 

            if 'Folds std' not in temp.columns:
                temp['Folds std'] = temp.std(axis=1)

            df_info = temp.join(trial.drop(['datetime_start', 'datetime_complete', 'state'], axis=1))
            df_info['duration'] = pd.to_timedelta(df_info['duration'], unit='s')
            df_info['duration'] = df_info['duration'].dt.total_seconds() / 60
            df_info.rename(columns={'duration': 'Duration (min)'}, inplace=True)

            df_info = df_info.sort_values(by='Folds mean', ascending=True)
            df_info.reset_index(inplace=True)
            df_info.index = [f'{i}º' for i in df_info.index + 1]
            df_info.rename_axis('Ranking', inplace=True)

            fist_level_multiindex = 'Categorical Crossentropy' if self.task == "classification" else "Mean Squared Error"

            trial_columns = [(fist_level_multiindex, col) for col in df_info.columns[: num_folds + 3]]
            hyperparameter_columns =[('Hyperparameters', col) for col in df_info.columns[num_folds + 3 :]]
            
            multi_columns = pd.MultiIndex.from_tuples(trial_columns + hyperparameter_columns)
            df_info.columns = multi_columns

            # df_info.style = df_info.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

            clear_output(wait=True)
            display(df_info)

        return sum(scores) / num_folds

    
    def hyperparameter_optimization(self, n_trials=1, **kwargs):

        num_folds = kwargs.get('num_folds', 5)
        columns_name = [f'Fold nº {i}' for i in range(1, num_folds + 1)]

        self.history_kfold = pd.DataFrame(columns=columns_name).rename_axis('Trial (nº)')

        directory = "../sqlite"
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.hyperparameter = optuna.create_study(
            study_name=f'optimization_{self.name}',
            storage=rf"sqlite:///{directory}/optimization_{self.name}.db",
            direction='minimize',
            load_if_exists=True
        )

        self.hyperparameter.optimize(lambda trial: self._optimizer(trial, **kwargs), n_trials = n_trials)

    def load(self, foldername, path="./saved"):

        if not os.path.exists(f'{path}/{foldername}'):
            print("There is no folder with that name!")
            return
        
        self.name = foldername
        self.preprocessor = load(f'{path}/{foldername}/preprocessor.joblib')

        try:
            self.model = XGBRegressor()
            self.model.load_model(f'{path}/{foldername}/model.bin')

        except:
            self.model = XGBClassifier()
            self.model.load_model(f'{path}/{foldername}/model.bin')

    
    def fit(self, return_history=False, graphic=False, graphic_save_extension=None, path="./saved", verbose=0, **kwargs):

        if self.hyperparameter is None:
           print("Realize a otimização de hiperparâmetros!")
           return

        self.model = self._make_xgBooster(
            tree_method = self.hyperparameter.best_params['tree_method'],
            booster = self.hyperparameter.best_params['booster'],
            learning_rate = self.hyperparameter.best_params['learning_rate'],
            min_split_loss = self.hyperparameter.best_params['min_split_loss'],
            max_depth = self.hyperparameter.best_params['max_depth'],
            min_child_weight = self.hyperparameter.best_params['min_child_weight'],
            max_delta_step = self.hyperparameter.best_params['max_delta_step'],
            subsample = self.hyperparameter.best_params['subsample'],
            sampling_method = self.hyperparameter.best_params['sampling_method'],
            colsample_bytree = self.hyperparameter.best_params['colsample_bytree'],
            colsample_bylevel = self.hyperparameter.best_params['colsample_bylevel'],
            colsample_bynode = self.hyperparameter.best_params['colsample_bynode'],
            reg_lambda = self.hyperparameter.best_params['reg_lambda'],
            reg_alpha = self.hyperparameter.best_params['reg_alpha'],
            scale_pos_weight = self.hyperparameter.best_params['scale_pos_weight'],
            grow_policy = self.hyperparameter.best_params['grow_policy'],
            max_leaves = self.hyperparameter.best_params['max_leaves'],
            max_bin = self.hyperparameter.best_params['max_bin'],
            num_parallel_tree = self.hyperparameter.best_params['num_parallel_tree']
        )
    
        self.model.fit(
            self.preprocessed_data['X_train'],
            self.preprocessed_data['y_train'],
            
            ## Não ocorre data leaking. EarlyStopping utiliza somente eval_set[-1]
            eval_set = [(
                self.preprocessed_data['X_train'],
                self.preprocessed_data['y_train']
                ), (
                self.preprocessed_data['X_test'],
                self.preprocessed_data['y_test']
                ), (
                self.preprocessed_data['X_val'],
                self.preprocessed_data['y_val']
                )],

            verbose=verbose
        )

        history = self.model.evals_result()

        if graphic:
            height = kwargs.get('subplot_height', 4)
            width = kwargs.get('subplot_width', 8)

            color = kwargs.get('subplot_color', {
                "train": 'red',
                "validation": "blue",
                "test": "green" 
            })

            if self.task == "regression":
                fig, axs = plt.subplots(len(self._metrics), 1, figsize=(width, height * (len(self._metrics))))

            else:
                fig, axs = plt.subplots(len(self._metrics) + 1, 1, figsize=(width, height * (len(self._metrics) + 1)))

            if not hasattr(axs, '__getitem__'):
                axs = [axs]
            
            title = "Bias-Variance Graphic (XG Boost)"

            fig.suptitle(title, fontweight='bold', fontsize=12)

            for i, metric in enumerate(self._metrics):

                if i == 0 and self.task == "regression":
                    y_true = self.preprocessed_data['y_test']
                    y_pred = self.model.predict(self.preprocessed_data['X_test'])

                    r2 = r2_score(y_true, y_pred)

                    axs[i].set_title(
                        f"R²: {r2:.3f} | {metric} (train: {history['validation_0'][metric][-1]:.5f}  val: {history['validation_2'][metric][-1]:.5f}  test: {history['validation_1'][metric][-1]:.5f})",
                        fontsize=12
                        )

                elif i == 0:
                    ## mlogloss == categorical crossentropy (muticlassification problem)
                    axs[i].set_title(
                        f"cost function [categorical crossentropy] (train: {history['validation_0'][metric][-1]:.5f}  val: {history['validation_2'][metric][-1]:.5f}  test: {history['validation_1'][metric][-1]:.5f})",
                        fontsize=12
                        )

                else:
                    axs[i].set_title(f"{metric} (train: {history['validation_0'][metric][-1]:.5f}  val: {history['validation_2'][metric][-1]:.5f}  test: {history['validation_1'][metric][-1]:.5f})")
                
                axs[i].plot(history['validation_0'][metric], linestyle='-', linewidth=2, label = 'Train', color=color['train'])
                axs[i].plot(history['validation_2'][metric], linestyle='-', linewidth=1, label = 'Validation', color=color['validation'])
                axs[i].axhline(y=history['validation_1'][metric][-1], linestyle='--', linewidth=1, label = 'Test', color=color['test'])
                axs[i].set_xlabel('Estimators')
                axs[i].set_ylabel('Metric')
                axs[i].legend(loc = 'best')
                axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))

            if self.task == "classification":
                y_true = self.preprocessed_data['y_test']                
                y_pred = self.model.predict(self.preprocessed_data['X_test'])

                if y_pred.ndim == 2:
                    y_pred = np.argmax(y_pred, axis=1)

                conf_mat = confusion_matrix(y_true, y_pred)

                encoder = self.preprocessor['target'].named_transformers_['target_preprocessor_cat'].named_steps['target_encoder_cat']
                class_labels = encoder.categories_[0].tolist()

                ax_conf = axs[-1]

                sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_conf,
                            xticklabels=class_labels, yticklabels=class_labels)
                
                ax_conf.set_xlabel(f'Predicted Values ({self.target})')
                ax_conf.set_ylabel(f'True Values ({self.target})')
                ax_conf.set_title('Confusion Matrix (Test Dataset)')

            plt.tight_layout(rect=[0, 0.05, 1, 0.98])

            if graphic_save_extension in ['png', 'svg', 'pdf', 'eps']:

                if not os.path.exists(f'{path}/{self.name}/figures'):
                    os.makedirs(f'{path}/{self.name}/figures')

                plt.savefig(f'{path}/{self.name}/figures/{title}.{graphic_save_extension}', format=f'{graphic_save_extension}')

            plt.show()
            plt.close()

        if return_history:
            return history

    
    def predict(self, x):
        _x = x.copy()
        _y_real = None

        if self.target in _x.columns:
            _y_real = _x[self.target]
            _x.drop(self.target, axis=1, inplace=True)

        ################### INFERENCE #######################
        start_time = time()

        _x_temp = self.preprocessor['features'].transform(_x)
        y = self.model.predict(_x_temp)

        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        if self.preprocessor.get('target') is not None:
            target_preprocessor = self.preprocessor['target'].named_transformers_['target_preprocessor_cat']
            y = target_preprocessor.inverse_transform(y.reshape(-1, 1)).reshape(-1)

        end_time = time() 
        #####################################################

        inference_time = end_time - start_time
        print(f"Inference time: {inference_time * 1000:.2f} milliseconds ({len(x)} register(s))") 

        if _y_real is not None:

            _x[self.target] = _y_real

        _x[f'{self.target} (XGB prediction)'] = y
        
        return _x

    def save(self, path="./saved"):

        if not os.path.exists(f'{path}/{self.name}'):
            os.makedirs(f'{path}/{self.name}')
        
        dump(self.preprocessor, f'{path}/{self.name}/preprocessor.joblib')
        self.model.save_model(f'{path}/{self.name}/model.bin')