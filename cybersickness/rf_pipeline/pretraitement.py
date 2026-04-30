import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def apply_target_discretization(df, target_profile):
    disc = target_profile.get("discretize")
    if disc is None:
        return df

    bins = disc["bins"]
    labels = disc["labels"]

    if len(labels) != len(bins) - 1:
        raise ValueError(
            f"'labels' doit avoir exactement {len(bins) - 1} elements pour {len(bins)} bornes. "
            f"Recu: {len(labels)} labels."
        )

    out = df.copy()
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

    if "n_valid_features" in out.columns:
        out = out[out["n_valid_features"] >= preprocess_profile.get("min_valid_features", 1)].copy()

    feature_cols = [
        c for c in out.columns if c not in {"target", "subject_id", "row_id", "window_start", "window_end"}
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
