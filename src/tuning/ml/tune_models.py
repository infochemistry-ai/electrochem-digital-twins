import os
import sys
import optuna
import numpy as np
import pandas as pd

from src.utils.paths import get_project_path
from src.tuning.ml.objectives import *


def tune_all_models():

    cur =  pd.read_csv(os.path.join(get_project_path(), 'data', 'FINAL_exp__cur.csv'))

    for i in ['classification', 'regression']:
        if i == 'classification':
            new_inh = cur.drop('ppm', axis=1)
            X_train, X_test, y_train, y_test = custom_split(new_inh)

            for m in ['catboost', 'mlp']:
                storage_path = os.path.join(get_project_path(), 'reports', 'optuna_db', m, i + '.db')
                study = optuna.create_study(
                study_name=f'cv_' + i + '_' + m,
                directions=['maximize'],
                load_if_exists=True,
                storage=f'sqlite:///{storage_path}')
                study.set_metric_names(['ROC_AUC'])

                if m == 'catboost':
                    study.optimize(lambda trial: catboost_class(trial, X_train, X_test, y_train, y_test), n_trials=500)

                elif m == 'mlp':
                    study.optimize(lambda trial: mlp_class(trial, X_train, X_test, y_train, y_test), n_trials=500)


        elif i == 'regression':
            new_inh = cur.drop('ppm', axis=1)
            data_n = {
                'benzotriazole': cur[cur['Inhibitor'] == 'benzotriazole'].drop('Inhibitor', axis=1), 
                '2-mercaptobenzimidazole': cur[cur['Inhibitor'] == '2-mercaptobenzimidazole'].drop('Inhibitor', axis=1), 
                'benzimidazole': cur[cur['Inhibitor'] == 'benzimidazole'].drop('Inhibitor', axis=1), 
                '4-benzylpiperidine': cur[cur['Inhibitor'] == '4-benzylpiperidine'].drop('Inhibitor', axis=1)
                }
            
            for name, data in data_n.items(): 
                X_train, X_test, y_train, y_test = custom_split(data)

                for j in ['catboost', 'mlp']:
                    storage_path = os.path.join(get_project_path(), 'reports', 'optuna_db', j, i + '.db')
                    study = optuna.create_study(
                        study_name=f'cv_{i}_{j}_{name}',
                        directions=['maximize', 'minimize', 'minimize'],
                        load_if_exists=True,
                        storage=f'sqlite:///{storage_path}')
                    study.set_metric_names(['R2', 'MAE', 'MSE'])

                    if j == 'catboost':
                        study.optimize(lambda trial: catboost_reg(trial, X_train, X_test, y_train, y_test), n_trials=500)

                    elif m == 'mlp':
                        study.optimize(lambda trial: mlp_class(trial, X_train, X_test, y_train, y_test), n_trials=500)


if __name__ == "__main__":
    tune_all_models()