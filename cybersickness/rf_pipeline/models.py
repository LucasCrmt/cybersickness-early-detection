from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid


def get_search_space(task_type, model_profile=None):
    common = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    if task_type == "classification":
        class_weight = "balanced" if model_profile is None else model_profile.get("class_weight", "balanced")
        common["class_weight"] = [class_weight]
    return common


def build_model(params, model_profile):
    if model_profile["task_type"] == "classification":
        return RandomForestClassifier(random_state=model_profile["random_state"], n_jobs=-1, **params)
    return RandomForestRegressor(random_state=model_profile["random_state"], n_jobs=-1, **params)


def run_hyperparam_search(X_train_imp, y_train, X_val_imp, y_val, model_profile):
    search_space = get_search_space(model_profile["task_type"], model_profile=model_profile)

    results = []
    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(search_space):
        model = build_model(params, model_profile)
        model.fit(X_train_imp, y_train)
        pred_val = model.predict(X_val_imp)

        row = deepcopy(params)
        if model_profile["task_type"] == "classification":
            row["val_accuracy"] = accuracy_score(y_val, pred_val)
            row["val_f1_weighted"] = f1_score(y_val, pred_val, average="weighted", zero_division=0)
            score = row["val_f1_weighted"]
        else:
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            row["val_rmse"] = rmse
            row["val_r2"] = r2_score(y_val, pred_val)
            score = -rmse

        results.append(row)
        if score > best_score:
            best_score = score
            best_params = deepcopy(params)

    results_df = pd.DataFrame(results)
    sort_col = "val_f1_weighted" if model_profile["task_type"] == "classification" else "val_rmse"
    ascending = model_profile["task_type"] != "classification"
    results_df = results_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
    return best_params, results_df


def fit_final_model(X_train_imp, y_train, X_val_imp, y_val, best_params, model_profile):
    X_train_val = np.vstack([X_train_imp, X_val_imp])
    y_train_val = np.concatenate([y_train, y_val])

    final_model = build_model(best_params, model_profile)
    final_model.fit(X_train_val, y_train_val)
    return final_model
