import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pipeline.extract import apply_sliding_window_sequences

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _resolve_filter_features(feature_cols, preprocess_profile, profile_key):
    selected = preprocess_profile.get(profile_key, "all")
    if isinstance(selected, str):
        if selected.strip().lower() != "all":
            raise ValueError(f"preprocess_profile['{profile_key}'] doit etre 'all' ou une liste de noms.")
        return list(feature_cols)

    if not isinstance(selected, (list, tuple, set)):
        raise ValueError(f"preprocess_profile['{profile_key}'] doit etre 'all' ou une liste de noms.")

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
            f"Les features suivantes demandees dans {profile_key} sont introuvables: "
            f"{missing}"
        )

    return resolved


def _resolve_temporal_filter_specs(feature_cols, preprocess_profile):
    specs = []

    if bool(preprocess_profile.get("apply_temporal_lowpass", False)):
        specs.append(
            {
                "name": "lowpass",
                "btype": "low",
                "order": int(preprocess_profile.get("lowpass_order", 4)),
                "cutoff_hz": float(preprocess_profile.get("lowpass_cutoff_hz", 0.05)),
                "min_points": int(preprocess_profile.get("lowpass_min_points", 16)),
                "cols": _resolve_filter_features(feature_cols, preprocess_profile, "lowpass_features"),
            }
        )

    if bool(preprocess_profile.get("apply_temporal_highpass", False)):
        specs.append(
            {
                "name": "highpass",
                "btype": "high",
                "order": int(preprocess_profile.get("highpass_order", 4)),
                "cutoff_hz": float(preprocess_profile.get("highpass_cutoff_hz", 0.01)),
                "min_points": int(preprocess_profile.get("highpass_min_points", 16)),
                "cols": _resolve_filter_features(feature_cols, preprocess_profile, "highpass_features"),
            }
        )

    if bool(preprocess_profile.get("apply_temporal_bandpass", False)):
        bandpass_ranges = preprocess_profile.get("bandpass_ranges_hz")
        if bandpass_ranges is None:
            # Compatibilite ascendante: une seule bande definie par low/high.
            bandpass_ranges = [
                [
                    float(preprocess_profile.get("bandpass_low_cutoff_hz", 0.01)),
                    float(preprocess_profile.get("bandpass_high_cutoff_hz", 0.2)),
                ]
            ]

        specs.append(
            {
                "name": "bandpass",
                "btype": "bandpass",
                "order": int(preprocess_profile.get("bandpass_order", 4)),
                "bands_hz": bandpass_ranges,
                "min_points": int(preprocess_profile.get("bandpass_min_points", 16)),
                "cols": _resolve_filter_features(feature_cols, preprocess_profile, "bandpass_features"),
            }
        )

    return specs


def _design_subject_filter(spec, nyquist):
    order = int(spec.get("order", 4))
    if order < 1:
        raise ValueError(f"preprocess_profile ordre invalide pour {spec.get('name')}: {order}.")

    btype = spec["btype"]
    eps = 0.95 * nyquist
    if btype in {"low", "high"}:
        cutoff_hz = float(spec["cutoff_hz"])
        if cutoff_hz <= 0:
            raise ValueError(f"Cutoff invalide pour {spec.get('name')}: {cutoff_hz}.")
        if btype == "low":
            cutoff_hz = min(cutoff_hz, eps)
            if cutoff_hz <= 0:
                return None, None
        else:
            if cutoff_hz >= nyquist:
                return None, None
        wn = cutoff_hz / nyquist
    elif btype == "bandpass":
        bands_hz = spec.get("bands_hz")
        if not isinstance(bands_hz, (list, tuple)) or len(bands_hz) == 0:
            raise ValueError("preprocess_profile['bandpass_ranges_hz'] doit contenir au moins une bande [low, high].")

        # Cas historique: un seul design de filtre si une unique bande est fournie.
        # Le cas multi-bandes est gere dans _apply_temporal_filters.
        first_band = bands_hz[0]
        if not isinstance(first_band, (list, tuple)) or len(first_band) != 2:
            raise ValueError("Chaque bande de bandpass_ranges_hz doit etre une paire [low_cutoff_hz, high_cutoff_hz].")

        low_cut = float(first_band[0])
        high_cut = float(first_band[1])
        if low_cut <= 0 or high_cut <= 0:
            raise ValueError("Les cutoffs bandpass doivent etre > 0.")
        high_cut = min(high_cut, eps)
        if not (low_cut < high_cut):
            return None, None
        wn = [low_cut / nyquist, high_cut / nyquist]
    else:
        raise ValueError(f"Type de filtre non supporte: {btype}")

    return butter(order, wn, btype=btype, analog=False)


def _design_bandpass_filters(spec, nyquist):
    """Construit une liste de filtres passe-bande valides a sommer pour un filtrage multi-bandes."""
    order = int(spec.get("order", 4))
    if order < 1:
        raise ValueError(f"preprocess_profile ordre invalide pour {spec.get('name')}: {order}.")

    bands_hz = spec.get("bands_hz")
    if not isinstance(bands_hz, (list, tuple)) or len(bands_hz) == 0:
        raise ValueError("preprocess_profile['bandpass_ranges_hz'] doit contenir au moins une bande [low, high].")

    eps = 0.95 * nyquist
    filters = []
    for band in bands_hz:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError("Chaque bande de bandpass_ranges_hz doit etre une paire [low_cutoff_hz, high_cutoff_hz].")

        low_cut = float(band[0])
        high_cut = float(band[1])
        if low_cut <= 0 or high_cut <= 0:
            raise ValueError("Les cutoffs bandpass doivent etre > 0.")

        high_cut = min(high_cut, eps)
        if not (low_cut < high_cut):
            # Ignore les bandes invalides plutot que bloquer tout le sujet.
            continue

        wn = [low_cut / nyquist, high_cut / nyquist]
        filters.append(butter(order, wn, btype="bandpass", analog=False))

    return filters


def _apply_temporal_filters(df, feature_cols, preprocess_profile):
    """Applique des filtres temporels par sujet (passe-bas, passe-haut, passe-bande)."""
    out = df.copy()

    approach = str(preprocess_profile.get("approach", "A")).strip().upper()
    if approach != "B":
        return out

    filter_specs = _resolve_temporal_filter_specs(feature_cols, preprocess_profile)
    if not filter_specs:
        return out

    time_col = preprocess_profile.get("time_col", "time")
    subject_col = preprocess_profile.get("subject_id_col", "subject_id")
    if time_col not in out.columns or subject_col not in out.columns:
        return out

    for spec in filter_specs:
        spec["min_points"] = max(int(spec.get("min_points", 16)), 8)
        spec["cols"] = [
            c for c in spec.get("cols", [])
            if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
        ]

    if not any(spec["cols"] for spec in filter_specs):
        return out

    for sid, idx in out.groupby(subject_col, sort=False, observed=True).groups.items():
        sid_idx = pd.Index(idx)
        block = out.loc[sid_idx, [time_col]].copy()
        block[time_col] = pd.to_numeric(block[time_col], errors="coerce")
        block = block.dropna(subset=[time_col]).sort_values(time_col)

        if block.empty:
            continue
        sorted_idx = block.index

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

        t_min, t_max = float(np.nanmin(t)), float(np.nanmax(t))
        n_grid = int(np.floor((t_max - t_min) / dt_med)) + 1
        if n_grid < 8:
            continue
        t_grid = np.linspace(t_min, t_max, n_grid)

        for spec in filter_specs:
            if not spec["cols"]:
                continue

            if spec["btype"] == "bandpass":
                bandpass_filters = _design_bandpass_filters(spec, nyquist)
                if len(bandpass_filters) == 0:
                    continue
                b = a = None
            else:
                b, a = _design_subject_filter(spec, nyquist)
                if b is None or a is None:
                    continue

            min_points = int(spec["min_points"])
            if len(t_grid) < min_points:
                continue

            for col in spec["cols"]:
                s = pd.to_numeric(out.loc[sorted_idx, col], errors="coerce")
                valid = s.notna().to_numpy()
                if int(valid.sum()) < min_points:
                    continue

                t_valid = t[valid]
                y_valid = s.to_numpy(dtype=float)[valid]
                if np.unique(t_valid).size < min_points:
                    continue

                y_grid = np.interp(t_grid, t_valid, y_valid)
                if spec["btype"] == "bandpass":
                    y_filt_grid = np.zeros_like(y_grid)
                    applied = False
                    for b_bp, a_bp in bandpass_filters:
                        bp_padlen = 3 * (max(len(a_bp), len(b_bp)) - 1)
                        if len(y_grid) <= bp_padlen:
                            continue
                        y_filt_grid += filtfilt(b_bp, a_bp, y_grid)
                        applied = True
                    if not applied:
                        continue
                else:
                    padlen = 3 * (max(len(a), len(b)) - 1)
                    if len(y_grid) <= padlen:
                        continue
                    y_filt_grid = filtfilt(b, a, y_grid)

                y_filt = np.interp(t_valid, t_grid, y_filt_grid)

                target_rows = sorted_idx[valid]
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

    def _resolve_exact_or_windowed_columns(raw_col_name):
        """Retourne soit une colonne exacte, soit un mapping step->colonne pour les colonnes fenetrees."""
        if raw_col_name in out.columns:
            return {"kind": "exact", "column": raw_col_name}

        key = str(raw_col_name).strip().lower()
        exact_matches = [c for c in out.columns if str(c).strip().lower() == key]
        if len(exact_matches) > 0:
            return {"kind": "exact", "column": exact_matches[0]}

        prefix = f"{key}__t"
        step_map = {}
        for c in out.columns:
            c_norm = str(c).strip().lower()
            if not c_norm.startswith(prefix):
                continue
            parts = str(c).rsplit("__t", 1)
            if len(parts) != 2:
                continue
            step_part = parts[1]
            if not str(step_part).isdigit():
                continue
            step_map[int(step_part)] = c

        if len(step_map) > 0:
            return {"kind": "windowed", "step_map": step_map}

        return {"kind": "missing", "column": raw_col_name}

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

        resolved = [_resolve_exact_or_windowed_columns(c) for c in cols_to_merge]
        kinds = {item["kind"] for item in resolved}

        if kinds == {"exact"}:
            resolved_cols = [item["column"] for item in resolved]
            subset = out[resolved_cols].apply(pd.to_numeric, errors="coerce")

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
            continue

        if kinds == {"windowed"}:
            step_sets = [set(item["step_map"].keys()) for item in resolved]
            common_steps = sorted(set.intersection(*step_sets)) if len(step_sets) > 0 else []
            if len(common_steps) == 0:
                raise ValueError(
                    f"column_aggregations['{result_col}']: aucune etape temporelle commune trouvee "
                    f"pour les colonnes {cols_to_merge}."
                )

            for step in common_steps:
                step_cols = [item["step_map"][step] for item in resolved]
                subset = out[step_cols].apply(pd.to_numeric, errors="coerce")
                target_col = f"{result_col}__t{int(step):04d}"

                if strategy == "mean":
                    out[target_col] = subset.mean(axis=1)
                elif strategy == "min":
                    out[target_col] = subset.min(axis=1)
                elif strategy == "max":
                    out[target_col] = subset.max(axis=1)
                elif strategy == "std":
                    out[target_col] = subset.std(axis=1)
                elif strategy == "sum":
                    out[target_col] = subset.sum(axis=1)
                else:
                    raise ValueError(
                        f"column_aggregations['{result_col}']: stratégie inconnue '{strategy}'. "
                        f"Stratégies supportées: mean, min, max, std, sum"
                    )
            continue

        missing = [item["column"] for item in resolved if item["kind"] == "missing"]
        if missing:
            raise ValueError(
                f"column_aggregations['{result_col}']: colonnes absentes: {missing}. "
                f"Colonnes disponibles: {list(out.columns)}"
            )

        raise ValueError(
            f"column_aggregations['{result_col}']: melange non supporte de colonnes brutes et fenetrees. "
            f"Colonnes demandees: {cols_to_merge}"
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
    base_excluded = {
        "target",
        "subject_id",
        "row_id",
        "time",
        "minute",
        "sampling_hz",
        "window_start",
        "window_end",
        "window_center",
        "window_id",
        "window_duration_s",
        "window_overlap_s",
    }
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

    # Approche B: filtres temporels optionnels sur la dimension temporelle.
    out = _apply_temporal_filters(out, feature_cols, preprocess_profile)

    # Fenetrage temporel post-pretraitement (par defaut), afin que include_features,
    # aggregations, clipping et filtrage s'appliquent sur les colonnes brutes.
    approach = str(preprocess_profile.get("approach", "A")).strip().upper()
    window_duration_s = preprocess_profile.get("window_duration_s")
    window_stage = str(preprocess_profile.get("window_stage", "post_preprocess")).strip().lower()
    if approach == "B" and window_duration_s is not None and window_stage != "extract":
        window_profile = dict(preprocess_profile)
        window_profile["window_feature_cols"] = list(feature_cols)
        out = apply_sliding_window_sequences(out, window_profile)

        # Reconstruit la liste de features fenetrees dans l'ordre step puis feature.
        step_map = {}
        for col in out.columns:
            if "__t" not in str(col):
                continue
            base, step_part = str(col).rsplit("__t", 1)
            if not str(step_part).isdigit():
                continue
            step = int(step_part)
            if base not in step_map:
                step_map[base] = set()
            step_map[base].add(step)

        windowed_feature_cols = []
        for step in sorted({s for steps in step_map.values() for s in steps}):
            for base in feature_cols:
                candidate = f"{base}__t{step:04d}"
                if candidate in out.columns:
                    windowed_feature_cols.append(candidate)
        feature_cols = windowed_feature_cols

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