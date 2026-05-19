import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _resolve_lowpass_features(feature_cols, preprocess_profile):
    selected = preprocess_profile.get("lowpass_features", "all")
    if isinstance(selected, str):
        if selected.strip().lower() != "all":
            raise ValueError("preprocess_profile['lowpass_features'] doit etre 'all' ou une liste de noms.")
        return list(feature_cols)

    if not isinstance(selected, (list, tuple, set)):
        raise ValueError("preprocess_profile['lowpass_features'] doit etre 'all' ou une liste de noms.")

    available_map = {str(c).strip().lower(): c for c in feature_cols}
    resolved = []
    missing = []
    seen = set()
    for raw_name in selected:
        key = str(raw_name).strip().lower()
        if key in available_map:
            col = available_map[key]
            if col not in seen:
                resolved.append(col)
                seen.add(col)
        else:
            missing.append(raw_name)

    if missing:
        raise ValueError(
            "Les features suivantes demandees dans lowpass_features sont introuvables: "
            f"{missing}"
        )

    return resolved


def _apply_temporal_lowpass_filter(df, feature_cols, preprocess_profile):
    """Applique un filtre passe-bas par sujet sur les features temporelles (approche B)."""
    out = df.copy()

    approach = str(preprocess_profile.get("approach", "A")).strip().upper()
    if approach != "B":
        return out
    if not bool(preprocess_profile.get("apply_temporal_lowpass", False)):
        return out

    time_col = preprocess_profile.get("time_col", "time")
    subject_col = preprocess_profile.get("subject_id_col", "subject_id")
    if time_col not in out.columns or subject_col not in out.columns:
        return out

    cutoff_hz = float(preprocess_profile.get("lowpass_cutoff_hz", 0.05))
    order = int(preprocess_profile.get("lowpass_order", 4))
    min_points = int(preprocess_profile.get("lowpass_min_points", 16))
    min_points = max(min_points, 8)

    if cutoff_hz <= 0:
        raise ValueError("preprocess_profile['lowpass_cutoff_hz'] doit etre > 0.")
    if order < 1:
        raise ValueError("preprocess_profile['lowpass_order'] doit etre >= 1.")

    lp_cols = [
        c for c in _resolve_lowpass_features(feature_cols, preprocess_profile)
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not lp_cols:
        return out

    for sid, idx in out.groupby(subject_col, sort=False, observed=True).groups.items():
        sid_idx = pd.Index(idx)
        block = out.loc[sid_idx, [time_col] + lp_cols].copy()
        block[time_col] = pd.to_numeric(block[time_col], errors="coerce")
        block = block.dropna(subset=[time_col]).sort_values(time_col)
        if len(block) < min_points:
            continue

        t = block[time_col].to_numpy(dtype=float)
        dt = np.diff(t)
        dt = dt[dt > 0]
        if dt.size == 0:
            continue

        dt_med = float(np.median(dt))
        if not np.isfinite(dt_med) or dt_med <= 0:
            continue

        fs_hz = 1.0 / dt_med
        nyquist = 0.5 * fs_hz
        if cutoff_hz >= nyquist:
            continue

        wn = cutoff_hz / nyquist
        b, a = butter(order, wn, btype="low", analog=False)

        t_min, t_max = float(np.nanmin(t)), float(np.nanmax(t))
        n_grid = int(np.floor((t_max - t_min) / dt_med)) + 1
        if n_grid < min_points:
            continue
        t_grid = np.linspace(t_min, t_max, n_grid)

        for col in lp_cols:
            s = pd.to_numeric(block[col], errors="coerce")
            valid = s.notna().to_numpy()
            if int(valid.sum()) < min_points:
                continue

            t_valid = t[valid]
            y_valid = s.to_numpy(dtype=float)[valid]
            if np.unique(t_valid).size < min_points:
                continue

            y_grid = np.interp(t_grid, t_valid, y_valid)
            padlen = 3 * (max(len(a), len(b)) - 1)
            if len(y_grid) <= padlen:
                continue

            y_filt_grid = filtfilt(b, a, y_grid)
            y_filt = np.interp(t_valid, t_grid, y_filt_grid)

            target_rows = block.index[valid]
            out.loc[target_rows, col] = y_filt

    return out


def apply_column_aggregations(df, preprocess_profile):
    """Fusionne plusieurs colonnes en une seule selon une stratégie d'agrégation.

    Permet par exemple de fusionner "Left Pupil Diameter" et "Right Pupil Diameter"
    en une seule colonne "pupil_diameter_avg" via stratégie "mean".

    Paramètres :
    df                : DataFrame
    preprocess_profile: dict de config, avec clé optionnelle 'column_aggregations'
        Clé attendue : 'column_aggregations' : dict de la forme
        {
            "nom_colonne_resultat": {
                "columns": ["col1", "col2"],
                "strategy": "mean" | "min" | "max" | "std" | "sum"
            },
            ...
        }

    Retour :
    DataFrame avec colonnes agrégées ajoutées (colonnes source conservées).
    Si 'column_aggregations' est absent, retourne df inchangé.
    """
    aggregations = preprocess_profile.get("column_aggregations")
    if aggregations is None or len(aggregations) == 0:
        return df

    out = df.copy()

    for result_col, spec in aggregations.items():
        cols_to_merge = spec.get("columns", [])
        strategy = spec.get("strategy", "mean")

        if len(cols_to_merge) == 0:
            raise ValueError(f"column_aggregations['{result_col}']: 'columns' est vide.")
        if len(cols_to_merge) < 2:
            raise ValueError(
                f"column_aggregations['{result_col}']: il faut au moins 2 colonnes. "
                f"Recu: {len(cols_to_merge)}"
            )

        missing = [c for c in cols_to_merge if c not in out.columns]
        if missing:
            raise ValueError(
                f"column_aggregations['{result_col}']: colonnes absentes: {missing}. "
                f"Colonnes disponibles: {list(out.columns)}"
            )

        # Extraire et convertir en numériques
        subset = out[cols_to_merge].apply(pd.to_numeric, errors="coerce")

        if strategy == "mean":
            out[result_col] = subset.mean(axis=1)
        elif strategy == "min":
            out[result_col] = subset.min(axis=1)
        elif strategy == "max":
            out[result_col] = subset.max(axis=1)
        elif strategy == "std":
            out[result_col] = subset.std(axis=1)
        elif strategy == "sum":
            out[result_col] = subset.sum(axis=1)
        else:
            raise ValueError(
                f"column_aggregations['{result_col}']: stratégie inconnue '{strategy}'. "
                f"Stratégies supportées: mean, min, max, std, sum"
            )

    return out


def apply_target_discretization(df, target_profile):
    out = df.copy()

    q = target_profile.get("clip_quantiles")
    if q is not None:
        q_low, q_high = q
        target_num = pd.to_numeric(out["target"], errors="coerce")
        lo = target_num.quantile(q_low)
        hi = target_num.quantile(q_high)
        out["target"] = target_num.clip(lo, hi)

    disc = target_profile.get("discretize")
    if disc is None:
        return out

    bins = disc["bins"]
    labels = disc["labels"]

    if len(labels) != len(bins) - 1:
        raise ValueError(
            f"'labels' doit avoir exactement {len(bins) - 1} elements pour {len(bins)} bornes. "
            f"Recu: {len(labels)} labels."
        )

    out["target"] = pd.cut(
        pd.to_numeric(out["target"], errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    out = out.dropna(subset=["target"])
    return out


def apply_preprocess(df, preprocess_profile):
    out = df.copy()

    # Agrégation de colonnes si configurée (avant tout autre traitement)
    out = apply_column_aggregations(out, preprocess_profile)

    if "n_valid_features" in out.columns:
        out = out[out["n_valid_features"] >= preprocess_profile.get("min_valid_features", 1)].copy()

    # Exclusions de base + exclusions configurees (insensibles a la casse)
    base_excluded = {"target", "subject_id", "row_id", "window_start", "window_end", "minute"}
    configured_excluded = set(preprocess_profile.get("exclude_features", []))
    excluded_norm = {str(c).strip().lower() for c in (base_excluded | configured_excluded)}

    feature_cols = [
        c for c in out.columns if str(c).strip().lower() not in excluded_norm
    ]

    q = preprocess_profile.get("clip_quantiles")
    if q is not None:
        q_low, q_high = q
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                lo = out[c].quantile(q_low)
                hi = out[c].quantile(q_high)
                out[c] = out[c].clip(lo, hi)

    if preprocess_profile.get("drop_low_information_features", False):
        nunique = out[feature_cols].nunique(dropna=False)
        drop_cols = nunique[nunique <= 1].index.tolist()
        out = out.drop(columns=drop_cols, errors="ignore")
        feature_cols = [c for c in feature_cols if c not in drop_cols]

    # Inclusion explicite des features: "all" ou liste de noms
    include_features = preprocess_profile.get("include_features", "all")
    if isinstance(include_features, str):
        if include_features.strip().lower() != "all":
            raise ValueError("preprocess_profile['include_features'] doit etre 'all' ou une liste de noms de features.")
    elif isinstance(include_features, (list, tuple, set)):
        available_map = {str(c).strip().lower(): c for c in feature_cols}
        selected = []
        missing = []
        seen = set()

        for raw_name in include_features:
            key = str(raw_name).strip().lower()
            if key in available_map:
                col = available_map[key]
                if col not in seen:
                    selected.append(col)
                    seen.add(col)
            else:
                missing.append(raw_name)

        if missing:
            raise ValueError(
                "Les features suivantes demandees dans include_features sont introuvables apres pretraitement: "
                f"{missing}"
            )
        feature_cols = selected
    else:
        raise ValueError("preprocess_profile['include_features'] doit etre 'all' ou une liste de noms de features.")

    # Approche B: filtre passe-bas optionnel sur la dimension temporelle.
    out = _apply_temporal_lowpass_filter(out, feature_cols, preprocess_profile)

    return out, feature_cols


def split_indices(y, groups, model_profile):
    idx = np.arange(len(y))
    split_method = model_profile["split_method"]
    test_size = model_profile["test_size"]
    val_size = model_profile["val_size"]
    random_state = model_profile["random_state"]
    is_classification = model_profile["task_type"] == "classification"

    if split_method == "random":
        train_val_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=y if is_classification else None
        )
        val_rel = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_rel,
            random_state=random_state,
            stratify=y[train_val_idx] if is_classification else None,
        )

    elif split_method == "group":
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_sub_idx, test_idx = next(gss_test.split(idx, y, groups))
        train_val_idx = idx[train_val_sub_idx]

        val_rel = val_size / (1 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=random_state)
        train_sub_idx, val_sub_idx = next(gss_val.split(train_val_idx, y[train_val_idx], groups[train_val_idx]))
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]

    else:
        raise ValueError("split_method doit etre 'random' ou 'group'.")

    return train_idx, val_idx, test_idx


def prepare_splits_and_impute(dataset_df, feature_cols, preprocess_profile, model_profile):
    X = dataset_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = dataset_df["target"].to_numpy()
    groups = dataset_df["subject_id"].astype(str).to_numpy()

    train_idx, val_idx, test_idx = split_indices(y, groups, model_profile)

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    imputer = SimpleImputer(strategy=preprocess_profile["imputation_strategy"])
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = None
    norm = preprocess_profile.get("normalization")
    if norm == "standard":
        scaler = StandardScaler()
    elif norm == "minmax":
        scaler = MinMaxScaler()

    if scaler is not None:
        X_train_imp = scaler.fit_transform(X_train_imp)
        X_val_imp = scaler.transform(X_val_imp)
        X_test_imp = scaler.transform(X_test_imp)

    return {
        "X_train_imp": X_train_imp,
        "X_val_imp": X_val_imp,
        "X_test_imp": X_test_imp,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "imputer": imputer,
        "scaler": scaler,
    }

def display_target_info(df, model_profile, target_profile=None, preprocess_profile=None):
    task = model_profile.get("task_type", "regression")
    target = df["target"]

    if task == "classification":
        print("Classes:", sorted(target.dropna().unique().tolist()))
        print("Distribution:")
        print(target.value_counts().sort_index())
    else:
        print("Distribution cible (regression):")
        print(target.describe())

    if target_profile and target_profile.get("clip_quantiles"):
        q_low, q_high = target_profile["clip_quantiles"]
        print(f"\nClip quantiles cible   : [{q_low}, {q_high}]")
    if preprocess_profile and preprocess_profile.get("clip_quantiles"):
        q_low, q_high = preprocess_profile["clip_quantiles"]
        print(f"Clip quantiles features: [{q_low}, {q_high}]")