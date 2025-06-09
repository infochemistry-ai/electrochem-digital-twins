import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostClassifier, CatBoostRegressor


def catboost_class(trial, X_train, X_test, y_train, y_test):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 1e1, log=True),
        'bootstrap_type': trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        'random_strength': trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0.0, 10.0),
        'od_type': trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        'od_wait': trial.suggest_int("od_wait", 10, 50),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'loss_function': 'MultiClass',
        'eval_metric': 'AUC',
        'task_type':'GPU',
        'devices': [0, 1]
    }
    
    model = CatBoostClassifier(**param, random_seed=42)
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    y_pred_proba = model.predict_proba(X_test)
    y_test_one_hot = pd.get_dummies(y_test)
    roc_auc = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class='ovr')
    
    return roc_auc


def catboost_reg(trial, X_train, X_test, y_train, y_test):
    params = {
        'iterations': trial.suggest_int("iterations", 1000, 5000),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        'depth': trial.suggest_int("depth", 1, 3),
        'bootstrap_type': trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        'random_strength': trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0.0, 10.0),
        'od_type': trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        'od_wait': trial.suggest_int("od_wait", 10, 50),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        'verbose': False,
        'task_type':'GPU',
        'devices': [0, 1]
    }
    
    cat = CatBoostRegressor(**params, random_state=42)
    cat.fit(X_train, y_train, verbose=False)
    
    y_pred = cat.predict(X_test)

    return (r2_score(y_test, y_pred), 
            mean_absolute_error(y_test, y_pred), 
            mean_squared_error(y_test, y_pred))


def mlp_class(trial, X_train, X_test, y_train, y_test):
    n_layers = trial.suggest_int('n_layers', 1, 10)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 967))

    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-8, 1e-1, log=True)
    max_iter = trial.suggest_int('max_iter', 200, 2000)
    beta_1 = trial.suggest_float('beta_1', 0.5, 0.9)
    beta_2 = trial.suggest_float('beta_2', 0.9, 0.9999)
    batch_size = trial.suggest_int('batch_size', 8, 128, step=8)
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])

    model = MLPClassifier(hidden_layer_sizes=tuple(layers), learning_rate_init=learning_rate_init, max_iter=max_iter,
                          beta_1=beta_1, beta_2=beta_2, batch_size=batch_size, solver='adam', activation=activation, random_state=42)

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)
    y_test_one_hot = pd.get_dummies(y_test)
    roc_auc = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class='ovr')

    return (roc_auc)


def mlp_reg(trial, X_train, X_test, y_train, y_test):
    n_layers = trial.suggest_int('n_layers', 1, 10)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 250))

    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-8, 1e-1, log=True)
    max_iter = trial.suggest_int('max_iter', 200, 2000)
    beta_1 = trial.suggest_float('beta_1', 0.5, 0.9)
    beta_2 = trial.suggest_float('beta_2', 0.9, 0.9999)
    batch_size = trial.suggest_int('batch_size', 8, 128, step=8)
    activation = trial.suggest_categorical('act_f', ['tanh', 'relu'])

    model = MLPRegressor(hidden_layer_sizes=tuple(layers), learning_rate_init=learning_rate_init, max_iter=max_iter,
                          beta_1=beta_1, beta_2=beta_2, batch_size=batch_size, solver='adam', activation=activation, random_state=42)

    model.fit(X_train, y_train)

    return (r2_score(y_test, model.predict(X_test)),
            mean_absolute_error(y_test, model.predict(X_test)),
            mean_squared_error(y_test, model.predict(X_test)))