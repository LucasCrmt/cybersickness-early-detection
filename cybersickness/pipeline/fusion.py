import numpy as np
from copy import deepcopy

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import ParameterGrid

from .models import _XGBClassifierWrapper, build_model, get_search_space


def split_feature_streams(feature_cols, fusion_profile):
    """Répartit feature_cols dans des flux par nom de colonne exact.

    Les colonnes déclarées dans streams qui n'existent pas dans feature_cols
    sont ignorées avec un avertissement.
    Les colonnes non assignées à aucun flux sont regroupées dans "other".

    Retourne {stream_name: [col_name, ...]}.
    """
    streams = fusion_profile["streams"]
    feature_set = set(feature_cols)
    result = {name: [] for name in streams}
    assigned = set()

    for stream_name, col_names in streams.items():
        for c in col_names:
            if c in feature_set:
                result[stream_name].append(c)
                assigned.add(c)
            else:
                print(f"[fusion] Avertissement : '{c}' introuvable dans feature_cols (flux '{stream_name}').")

    unassigned = [c for c in feature_cols if c not in assigned]
    if unassigned:
        result["other"] = unassigned

    return result


def _col_indices(stream_cols, feature_cols):
    col_list = list(feature_cols)
    return [col_list.index(c) for c in stream_cols]


def train_stream_models(X_train, y_train, X_val, y_val, stream_map, feature_cols, model_profile):
    """Entraîne un modèle XGBoost par flux via grid search sur le val set.

    Les stream_models sont entraînés sur train uniquement — leurs prédictions
    sur val serviront à entraîner le méta-modèle sans fuite de données.

    Retourne {stream_name: fitted_model}.
    """
    is_classif = model_profile["task_type"] == "classification"
    search_space = get_search_space(model_profile["task_type"], model_profile)
    fitted = {}

    for stream_name, stream_cols in stream_map.items():
        if not stream_cols:
            print(f"[fusion] Stream '{stream_name}' vide, ignoré.")
            continue

        idx = _col_indices(stream_cols, feature_cols)
        X_tr = X_train[:, idx]
        X_vl = X_val[:, idx]

        best_score, best_model = -np.inf, None
        for params in ParameterGrid(search_space):
            m = build_model(params, model_profile)
            m.fit(X_tr, y_train)
            pred = m.predict(X_vl)
            score = (
                f1_score(y_val, pred, average="weighted", zero_division=0)
                if is_classif
                else -float(np.sqrt(mean_squared_error(y_val, pred)))
            )
            if score > best_score:
                best_score, best_model = score, deepcopy(m)

        fitted[stream_name] = best_model
        label = "F1" if is_classif else "RMSE"
        val = best_score if is_classif else -best_score
        print(f"[fusion] Stream '{stream_name}' ({len(stream_cols)} features) — {label} val: {val:.4f}")

    return fitted


def make_meta_features(stream_models, stream_map, X, feature_cols, is_classification):
    """Concatène les prédictions de chaque flux en un vecteur méta-feature.

    Classification : predict_proba (vecteur de probabilités par classe).
    Régression     : predict (scalaire → colonne).
    """
    parts = []
    for stream_name, model in stream_models.items():
        idx = _col_indices(stream_map[stream_name], feature_cols)
        X_s = X[:, idx]
        if is_classification and hasattr(model, "predict_proba"):
            parts.append(model.predict_proba(X_s))
        else:
            parts.append(model.predict(X_s).reshape(-1, 1))
    return np.hstack(parts)


def build_meta_model(fusion_profile, task_type, seed=42):
    """Construit le méta-modèle (dernière couche de fusion).

    Options meta_model : logistic_regression / random_forest / svm / xgboost
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from xgboost import XGBRegressor

    meta_type = fusion_profile.get("meta_model", "logistic_regression")
    is_classif = task_type == "classification"

    if meta_type == "random_forest":
        if is_classif:
            return RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1)
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)

    if meta_type == "svm":
        if is_classif:
            return SVC(kernel="rbf", class_weight="balanced")
        return SVR(kernel="rbf")

    if meta_type == "xgboost":
        if is_classif:
            return _XGBClassifierWrapper(n_estimators=100, random_state=seed, eval_metric="logloss")
        return XGBRegressor(n_estimators=100, random_state=seed)

    # logistic_regression (defaut classif) / ridge (defaut regression)
    if is_classif:
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    return Ridge()
