from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

_VALID_MODEL_TYPES = {"random_forest", "xgboost", "svm"}


def _get_model_type(model_profile):
    mt = model_profile.get("model_type", "random_forest").lower()
    if mt not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type invalide: '{mt}'. Valeurs acceptées: {_VALID_MODEL_TYPES}")
    return mt


def get_search_space(task_type, model_profile=None):
    mt = _get_model_type(model_profile or {})

    if mt == "xgboost":
        return {
            "n_estimators": [100, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

    if mt == "svm":
        return {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }

    # random_forest
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
    mt = _get_model_type(model_profile)
    is_classif = model_profile["task_type"] == "classification"

    if mt == "xgboost":
        if is_classif:
            return XGBClassifier(random_state=model_profile["random_state"], n_jobs=-1, eval_metric="logloss", **params)
        return XGBRegressor(random_state=model_profile["random_state"], n_jobs=-1, **params)
    if mt == "random_forest":
        if is_classif:
            return RandomForestClassifier(random_state=model_profile["random_state"], n_jobs=-1, **params)
        return RandomForestRegressor(random_state=model_profile["random_state"], n_jobs=-1, **params)

    if mt == "svm":
        class_weight = model_profile.get("class_weight", None) if is_classif else None
        if is_classif:
            return SVC(class_weight=class_weight, **params)
        return SVR(**params)



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
