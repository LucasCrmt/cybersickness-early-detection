"""Fonctions de segmentation de séries temporelles brutes pour la fusion multi-flux.

Ce module contient les utilitaires spécifiques au pipeline raw (Fusion_raw.ipynb) :
- segment_sequences_by_minute  : découpe par fenêtres d'une minute
- segment_sequences_sliding_window : découpe par fenêtres glissantes
- prepare_splits_and_impute_3d : split/imputation/normalisation sur tableaux 3D
"""
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .pretraitement import split_indices


def segment_sequences_by_minute(raw_df, feature_cols, target_df,
                                T=60, subject_col="subject_id", time_col="time", pad_value=0.0):
    """Segmente un DataFrame brut en fenêtres (sujet, minute) de T échantillons.

    Chaque minute de signal brut est rééchantillonnée par interpolation linéaire
    vers exactement T pas de temps uniformément espacés dans [0, 60).

    Args:
        raw_df      : DataFrame long [subject_col, time_col, *feature_cols]
        feature_cols: liste de colonnes à extraire
        target_df   : DataFrame [subject_col, 'minute', 'target']
        T           : nombre de pas de temps par segment
        subject_col : nom de la colonne sujet dans raw_df
        time_col    : nom de la colonne temps (en secondes) dans raw_df
        pad_value   : valeur de remplissage si signal absent

    Returns:
        X      : np.ndarray (n_segments, T, n_features) float32
        y      : np.ndarray (n_segments,) object
        groups : np.ndarray (n_segments,) str — identifiants sujets
    """
    target_df = target_df.copy()
    target_df[subject_col] = target_df[subject_col].astype(str).str.strip()

    target_lookup = {}
    for _, row in target_df.iterrows():
        key = (str(row[subject_col]).strip(), int(row["minute"]))
        val = row["target"]
        if not pd.isna(val):
            target_lookup[key] = val

    raw_df = raw_df.copy()
    raw_df[subject_col] = raw_df[subject_col].astype(str).str.strip()
    raw_df["_minute"] = (pd.to_numeric(raw_df[time_col], errors="coerce") // 60).astype("Int64")
    raw_df = raw_df.dropna(subset=["_minute"])

    missing_cols = [c for c in feature_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"segment_sequences_by_minute: colonnes absentes dans raw_df: {missing_cols}")

    t_grid = np.linspace(0.0, 60.0, T, endpoint=False)
    X_list, y_list, groups_list = [], [], []

    for (subj, minute), group in raw_df.groupby([subject_col, "_minute"], sort=True, observed=True):
        key = (str(subj).strip(), int(minute))
        if key not in target_lookup:
            continue

        group = group.sort_values(time_col)
        t_raw = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float) % 60.0
        segment = np.full((T, len(feature_cols)), pad_value, dtype=np.float32)

        for j, col in enumerate(feature_cols):
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                if valid.sum() == 1:
                    segment[:, j] = vals[valid][0]
                continue
            t_v = t_raw[valid]
            v_v = vals[valid]
            order = np.argsort(t_v)
            t_v, v_v = t_v[order], v_v[order]
            _, ui = np.unique(t_v, return_index=True)
            t_v, v_v = t_v[ui], v_v[ui]
            if len(t_v) < 2:
                segment[:, j] = v_v[0]
                continue
            t_clip = np.clip(t_grid, t_v[0], t_v[-1])
            segment[:, j] = np.interp(t_clip, t_v, v_v).astype(np.float32)

        X_list.append(segment)
        y_list.append(target_lookup[key])
        groups_list.append(str(subj).strip())

    if not X_list:
        raise ValueError("segment_sequences_by_minute: aucun segment construit. "
                         "Verifier la correspondance sujet/minute entre raw_df et target_df.")

    return np.stack(X_list), np.array(y_list), np.array(groups_list)


def segment_sequences_sliding_window(raw_df, feature_cols, target_df,
                                     window_s=20.0, stride_s=10.0, T=200,
                                     subject_col="subject_id", time_col="time",
                                     pad_value=0.0,
                                     session_min_s=None, session_max_s=None):
    """Segmente un DataFrame en fenêtres glissantes de window_s secondes.

    Chaque fenêtre est rééchantillonnée à T points par interpolation linéaire.
    Le label est déterminé par floor(t_centre / 60) pour trouver la minute
    correspondante dans target_df.

    Les fenêtres dont le centre tombe dans une minute sans label sont ignorées.

    Args:
        raw_df        : DataFrame long [subject_col, time_col, *feature_cols]
        feature_cols  : liste de colonnes à extraire
        target_df     : DataFrame [subject_col, 'minute', 'target']
        window_s      : durée de la fenêtre en secondes
        stride_s      : décalage entre fenêtres en secondes
        T             : nombre de timesteps par fenêtre (rééchantillonnage uniforme)
        subject_col   : colonne identifiant le sujet
        time_col      : colonne de temps en secondes
        pad_value     : valeur de remplissage si données insuffisantes
        session_min_s : si défini, n'inclut que les fenêtres dont le centre >= session_min_s
        session_max_s : si défini, n'inclut que les fenêtres dont le centre <  session_max_s

    Returns:
        X      : np.ndarray (n_windows, T, n_features) float32
        y      : np.ndarray (n_windows,) object
        groups : np.ndarray (n_windows,) str
    """
    target_df = target_df.copy()
    target_df[subject_col] = target_df[subject_col].astype(str).str.strip()

    target_lookup = {}
    for _, row in target_df.iterrows():
        key = (str(row[subject_col]).strip(), int(row["minute"]))
        val = row["target"]
        if not pd.isna(val):
            target_lookup[key] = val

    raw_df = raw_df.copy()
    raw_df[subject_col] = raw_df[subject_col].astype(str).str.strip()
    raw_df[time_col] = pd.to_numeric(raw_df[time_col], errors="coerce")
    raw_df = raw_df.dropna(subset=[time_col]).sort_values([subject_col, time_col])

    missing_cols = [c for c in feature_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"segment_sequences_sliding_window: colonnes absentes: {missing_cols}")

    n_features = len(feature_cols)
    t_local = np.linspace(0.0, window_s, T, endpoint=False)
    X_list, y_list, groups_list = [], [], []

    for sid, grp in raw_df.groupby(subject_col, sort=True, observed=True):
        grp = grp.reset_index(drop=True)
        t_abs = grp[time_col].to_numpy(dtype=float)
        vals_all = grp[feature_cols].to_numpy(dtype=float)

        t_min = t_abs[0]
        t_max = t_abs[-1]
        if t_max - t_min < window_s:
            continue

        ws = t_min
        while ws + window_s <= t_max + 1e-9:
            we = ws + window_s
            t_center = ws + window_s / 2.0

            if session_min_s is not None and t_center < session_min_s:
                ws += stride_s
                continue
            if session_max_s is not None and t_center >= session_max_s:
                ws += stride_s
                continue

            minute = int(np.floor(t_center / 60.0))
            key = (str(sid).strip(), minute)

            if key in target_lookup:
                mask = (t_abs >= ws) & (t_abs < we)
                t_win = t_abs[mask]

                if len(t_win) >= 2:
                    t_rel = t_win - ws
                    vals_win = vals_all[mask]
                    segment = np.full((T, n_features), pad_value, dtype=np.float32)

                    for j in range(n_features):
                        v = vals_win[:, j]
                        valid = ~np.isnan(v)
                        if valid.sum() < 2:
                            if valid.sum() == 1:
                                segment[:, j] = v[valid][0]
                            continue
                        t_v = t_rel[valid]
                        v_v = v[valid]
                        order = np.argsort(t_v)
                        t_v, v_v = t_v[order], v_v[order]
                        _, ui = np.unique(t_v, return_index=True)
                        t_v, v_v = t_v[ui], v_v[ui]
                        if len(t_v) < 2:
                            segment[:, j] = v_v[0]
                            continue
                        t_clip = np.clip(t_local, t_v[0], t_v[-1])
                        segment[:, j] = np.interp(t_clip, t_v, v_v).astype(np.float32)

                    X_list.append(segment)
                    y_list.append(target_lookup[key])
                    groups_list.append(str(sid).strip())

            ws += stride_s

    if not X_list:
        raise ValueError("segment_sequences_sliding_window: aucun segment construit. "
                         "Verifier la correspondance sujet/minute entre raw_df et target_df.")

    return np.stack(X_list), np.array(y_list), np.array(groups_list)


def prepare_splits_and_impute_3d(X, y, groups, preprocess_profile, model_profile):
    """Split et normalise des tableaux 3D (n_segments, T, n_features).

    L'imputation et la normalisation sont appliquées par feature (sur l'axe feature),
    en aplatissant les dimensions (n_segments, T) ensemble pour calculer les stats.

    Returns le même dictionnaire que prepare_splits_and_impute.
    """
    train_idx, val_idx, test_idx = split_indices(y, groups, model_profile)

    n_features = X.shape[2]
    T = X.shape[1]

    X_train = X[train_idx].copy()
    X_val   = X[val_idx].copy()
    X_test  = X[test_idx].copy()

    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    imputer = SimpleImputer(strategy=preprocess_profile.get("imputation_strategy", "median"))
    X_tr_2d = X_train.reshape(-1, n_features)
    X_vl_2d = X_val.reshape(-1, n_features)
    X_te_2d = X_test.reshape(-1, n_features)

    X_train = imputer.fit_transform(X_tr_2d).reshape(len(train_idx), T, n_features)
    X_val   = imputer.transform(X_vl_2d).reshape(len(val_idx),   T, n_features)
    X_test  = imputer.transform(X_te_2d).reshape(len(test_idx),  T, n_features)

    scaler = None
    norm = preprocess_profile.get("normalization", "standard")
    if norm == "standard":
        scaler = StandardScaler()
    elif norm == "minmax":
        scaler = MinMaxScaler()

    if scaler is not None:
        X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(len(train_idx), T, n_features)
        X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(len(val_idx),   T, n_features)
        X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(len(test_idx),  T, n_features)

    return {
        "X_train_imp": X_train,
        "X_val_imp":   X_val,
        "X_test_imp":  X_test,
        "y_train":     y_train,
        "y_val":       y_val,
        "y_test":      y_test,
        "train_idx":   train_idx,
        "val_idx":     val_idx,
        "test_idx":    test_idx,
        "imputer":     imputer,
        "scaler":      scaler,
    }
