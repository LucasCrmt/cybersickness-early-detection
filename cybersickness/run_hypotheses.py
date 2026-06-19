import argparse
import collections
import copy
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.evaluation import (
    evaluate_test_set,
    train_with_optional_hyperparameter_search,
)
from pipeline.extract import add_target, load_features_for_approach, load_per_minute_targets
from pipeline.fusion import (
    build_meta_model,
    make_meta_features,
    split_feature_streams,
    train_stream_models,
)
from pipeline.pretraitement import (
    apply_column_aggregations,
    apply_preprocess,
    apply_target_discretization,
    prepare_splits_and_impute,
)
from pipeline.raw_segmentation import prepare_splits_and_impute_3d, segment_sequences_sliding_window
from pipeline.reporting import export_visual_report


DEEP_MODELS = {"cnn_1d", "inception_time", "bilstm", "cnn_lstm", "multistream"}


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _slugify(text):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(text).strip().lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "hypothesis"


def _resolve_path(path_value, repo_root):
    if isinstance(path_value, (list, tuple)):
        resolved = []
        for pv in path_value:
            path = Path(pv)
            resolved.append(str(path if path.is_absolute() else (repo_root / path).resolve()))
        return resolved
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _resolve_config_path(path_value, script_dir, repo_root):
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [
        (Path.cwd() / path).resolve(),
        (script_dir / path).resolve(),
        (repo_root / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _reshape_for_deep_models(X_train_imp, X_val_imp, X_test_imp, feature_cols_train):
    windowed_feature_cols = [c for c in feature_cols_train if "__t" in str(c)]
    if windowed_feature_cols:
        base_feature_order = []
        step_numbers = []
        for col in windowed_feature_cols:
            base_name, step_part = str(col).rsplit("__t", 1)
            if base_name not in base_feature_order:
                base_feature_order.append(base_name)
            try:
                step_numbers.append(int(step_part))
            except ValueError:
                pass

        step_numbers = sorted(set(step_numbers))
        sequence_length = len(step_numbers)
        n_features = len(base_feature_order)

        ordered_window_cols = [
            f"{feat}__t{step:04d}"
            for step in step_numbers
            for feat in base_feature_order
        ]
        ordered_indices = [
            feature_cols_train.index(col)
            for col in ordered_window_cols
            if col in feature_cols_train
        ]

        if len(ordered_indices) != len(ordered_window_cols):
            missing = [col for col in ordered_window_cols if col not in feature_cols_train]
            raise ValueError(f"Missing window columns: {missing}")

        X_train_imp = X_train_imp[:, ordered_indices]
        X_val_imp = X_val_imp[:, ordered_indices]
        X_test_imp = X_test_imp[:, ordered_indices]

        X_train_imp = X_train_imp.reshape((-1, sequence_length, n_features))
        X_val_imp = X_val_imp.reshape((-1, sequence_length, n_features))
        X_test_imp = X_test_imp.reshape((-1, sequence_length, n_features))
    else:
        sequence_length = X_train_imp.shape[1]
        n_features = 1
        X_train_imp = X_train_imp.reshape((-1, sequence_length, n_features))
        X_val_imp = X_val_imp.reshape((-1, sequence_length, n_features))
        X_test_imp = X_test_imp.reshape((-1, sequence_length, n_features))

    return X_train_imp, X_val_imp, X_test_imp, int(sequence_length), int(n_features)


def _prepare_profiles(repo_root, base_profiles, hypothesis_profiles):
    profiles = _deep_merge(base_profiles, hypothesis_profiles)

    for key in ["data_profile", "preprocess_profile", "target_profile", "model_profile", "output_profile"]:
        if key not in profiles:
            raise ValueError(f"Missing profile in config: {key}")

    data_profile = profiles["data_profile"]
    target_profile = profiles["target_profile"]
    output_profile = profiles["output_profile"]

    if data_profile.get("source") == "csv" and "file_path" in data_profile:
        resolved = _resolve_path(data_profile["file_path"], repo_root)
        data_profile["file_path"] = resolved if isinstance(resolved, list) else str(resolved)
    if "file_path_phase2" in data_profile:
        data_profile["file_path_phase2"] = str(_resolve_path(data_profile["file_path_phase2"], repo_root))

    if target_profile.get("source") == "csv" and "csv_path" in target_profile:
        resolved = _resolve_path(target_profile["csv_path"], repo_root)
        target_profile["csv_path"] = resolved if isinstance(resolved, list) else str(resolved)
    if target_profile.get("source") == "xlsx" and "xlsx_path" in target_profile:
        resolved = _resolve_path(target_profile["xlsx_path"], repo_root)
        target_profile["xlsx_path"] = resolved if isinstance(resolved, list) else str(resolved)
    if "xlsx_path_phase2" in target_profile:
        target_profile["xlsx_path_phase2"] = str(_resolve_path(target_profile["xlsx_path_phase2"], repo_root))

    if "output_dir" in output_profile:
        output_profile["output_dir"] = str(_resolve_path(output_profile["output_dir"], repo_root))

    return profiles


def _run_standard(name, profiles):
    """Pipeline standard (approche A/B tabulaire, features 2D)."""
    data_profile = profiles["data_profile"]
    preprocess_profile = profiles["preprocess_profile"]
    target_profile = profiles["target_profile"]
    model_profile = profiles["model_profile"]
    output_profile = profiles["output_profile"]

    features_df = load_features_for_approach(
        data_profile=data_profile,
        preprocess_profile=preprocess_profile,
        verbose=True,
    )

    dataset_df = add_target(features_df, target_profile)

    task_type = str(model_profile.get("task_type", "classification")).lower()
    if task_type == "classification":
        if pd.api.types.is_numeric_dtype(dataset_df["target"]):
            dataset_df = apply_target_discretization(dataset_df, target_profile)

    raw_df = dataset_df.copy()
    already_windowed_input = any("__t" in str(c) for c in dataset_df.columns)

    if not already_windowed_input:
        preprocess_profile_visu = {
            **preprocess_profile,
            "window_duration_s": None,
        }
        preprocess_profile_visu.pop("window_overlap_s", None)
        preprocess_profile_visu.pop("window_overlap_ratio", None)
        dataset_df_visu, feature_cols_visu = apply_preprocess(dataset_df.copy(), preprocess_profile_visu)
        dataset_df_train, feature_cols_train = apply_preprocess(dataset_df.copy(), preprocess_profile)
    else:
        dataset_df_train = dataset_df.copy()
        feature_cols_train = [c for c in dataset_df_train.columns if "__t" in str(c)]
        dataset_df_visu = dataset_df_train.copy()
        feature_cols_visu = feature_cols_train.copy()

    prepared = prepare_splits_and_impute(
        dataset_df=dataset_df_train,
        feature_cols=feature_cols_train,
        preprocess_profile=preprocess_profile,
        model_profile=model_profile,
    )

    X_train_imp = prepared["X_train_imp"]
    X_val_imp = prepared["X_val_imp"]
    X_test_imp = prepared["X_test_imp"]
    y_train = prepared["y_train"]
    y_val = prepared["y_val"]
    y_test = prepared["y_test"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    test_idx = prepared["test_idx"]

    if task_type == "regression":
        y_train = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
        y_val = pd.to_numeric(pd.Series(y_val),   errors="coerce").to_numpy(dtype=float)
        y_test = pd.to_numeric(pd.Series(y_test),  errors="coerce").to_numpy(dtype=float)
        if np.isnan(y_train).any() or np.isnan(y_val).any() or np.isnan(y_test).any():
            raise ValueError("Regression target contains NaN after numeric conversion.")

    sequence_length = None
    n_features = None
    model_type = str(model_profile.get("model_type", "random_forest")).lower()
    if model_type in DEEP_MODELS:
        X_train_imp, X_val_imp, X_test_imp, sequence_length, n_features = _reshape_for_deep_models(
            X_train_imp, X_val_imp, X_test_imp, feature_cols_train
        )
        model_profile["n_classes"] = int(pd.Series(y_train).nunique()) if task_type == "classification" else 1
        model_profile["sequence_length"] = int(sequence_length)
        model_profile["n_features"] = int(n_features)

    final_model, best_params, _ = train_with_optional_hyperparameter_search(
        X_train_imp=X_train_imp,
        y_train=y_train,
        X_val_imp=X_val_imp,
        y_val=y_val,
        model_profile=model_profile,
        approach=preprocess_profile.get("approach"),
        sequence_length=sequence_length,
        n_features=n_features,
        verbose=True,
    )

    _, metrics, _ = evaluate_test_set(
        final_model=final_model,
        X_test_imp=X_test_imp,
        y_test=y_test,
        model_profile=model_profile,
        target_profile=target_profile,
        show_plots=False,
    )

    report_info = None
    if bool(output_profile.get("save_visual_report", True)):
        report_info = export_visual_report(
            dataset_df=dataset_df_visu,
            feature_cols=feature_cols_visu,
            model_profile=model_profile,
            output_profile=output_profile,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            target_profile=target_profile,
            raw_df=raw_df,
            preprocess_profile=preprocess_profile_visu if not already_windowed_input else preprocess_profile,
            best_params=best_params,
            metrics=metrics,
            final_model=final_model,
            X_test_imp=X_test_imp,
            y_test=y_test,
            dataset_df_train=dataset_df_train,
            feature_cols_train=feature_cols_train,
        )

    return metrics, report_info


def _run_raw(name, profiles):
    """Pipeline segmentation brute (approche B raw, arrays 3D, ex-run_fusion_hypotheses)."""
    data_profile       = profiles["data_profile"]
    preprocess_profile = profiles["preprocess_profile"]
    target_profile     = profiles["target_profile"]
    model_profile      = profiles["model_profile"]
    raw_profile        = profiles["raw_profile"]
    output_profile     = profiles["output_profile"]

    features_df = load_features_for_approach(
        data_profile=data_profile,
        preprocess_profile=preprocess_profile,
        verbose=True,
    )
    features_df = apply_column_aggregations(features_df, preprocess_profile)

    if raw_profile.get("per_subject_norm", False):
        norm_cols = [
            c for c in preprocess_profile.get("include_features", [])
            if c in features_df.columns and pd.api.types.is_numeric_dtype(features_df[c])
        ]
        if "Pupil diameter AVG" in features_df.columns and "Pupil diameter AVG" not in norm_cols:
            norm_cols.append("Pupil diameter AVG")
        for sid, idx in features_df.groupby("subject_id", sort=False).groups.items():
            sub   = features_df.loc[idx, norm_cols].astype(float)
            mu    = sub.mean()
            sigma = sub.std().replace(0, 1e-8)
            features_df.loc[idx, norm_cols] = (sub - mu) / sigma
        print(f"Normalisation per-sujet : {features_df['subject_id'].nunique()} sujets, {len(norm_cols)} features")

    target_df = load_per_minute_targets(target_profile)
    target_df["subject_id"] = target_df["subject_id"].astype(str).str.strip()
    target_df = apply_target_discretization(target_df, target_profile)
    print("Distribution cibles :")
    for k, v in sorted(collections.Counter(target_df["target"].tolist()).items()):
        print(f"  {k}: {v}")

    proc_df, feature_cols = apply_preprocess(features_df.copy(), preprocess_profile)

    window_s      = float(raw_profile["window_s"])
    stride_s      = float(raw_profile["stride_s"])
    T             = int(raw_profile["sequence_length"])
    session_min_s = raw_profile.get("session_min_s")
    session_max_s = raw_profile.get("session_max_s")

    if session_min_s is not None or session_max_s is not None:
        lo = f"{session_min_s}s" if session_min_s is not None else "0"
        hi = f"{session_max_s}s" if session_max_s is not None else "fin"
        print(f"Segmentation - fenetre {window_s}s, stride {stride_s}s, T={T}, plage [{lo}, {hi}[ ...")
    else:
        print(f"Segmentation - fenetre {window_s}s, stride {stride_s}s, T={T} ...")

    X_raw, y_raw, groups_raw = segment_sequences_sliding_window(
        raw_df=proc_df,
        feature_cols=feature_cols,
        target_df=target_df,
        window_s=window_s,
        stride_s=stride_s,
        T=T,
        subject_col="subject_id",
        time_col="time",
        pad_value=float(raw_profile.get("pad_value", 0.0)),
        session_min_s=float(session_min_s) if session_min_s is not None else None,
        session_max_s=float(session_max_s) if session_max_s is not None else None,
    )
    print(f"X shape : {X_raw.shape}  (n_fenetres, T, n_features)")

    model_profile["n_classes"]       = int(len(np.unique(y_raw)))
    model_profile["sequence_length"] = T
    model_profile["n_features"]      = int(len(feature_cols))

    data_source = str(data_profile.get("data_source", "phase1")).strip().lower()
    use_root_grouping = data_source == "both" or bool(model_profile.get("group_split_by_root_subject", False))
    groups_for_split = (
        np.array([re.sub(r"_P\d+$", "", g) for g in groups_raw], dtype=object)
        if use_root_grouping else groups_raw
    )

    prepared = prepare_splits_and_impute_3d(
        X=X_raw,
        y=y_raw,
        groups=groups_for_split,
        preprocess_profile=preprocess_profile,
        model_profile=model_profile,
    )
    X_train_imp = prepared["X_train_imp"]
    X_val_imp   = prepared["X_val_imp"]
    X_test_imp  = prepared["X_test_imp"]
    y_train     = prepared["y_train"]
    y_val       = prepared["y_val"]
    y_test      = prepared["y_test"]
    train_idx   = prepared["train_idx"]
    val_idx     = prepared["val_idx"]
    test_idx    = prepared["test_idx"]
    print(f"Split train/val/test : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    dataset_df = pd.DataFrame(X_raw.mean(axis=1), columns=feature_cols)
    dataset_df["target"]     = y_raw
    dataset_df["subject_id"] = groups_raw
    raw_df = dataset_df.copy()

    final_model, best_params, _ = train_with_optional_hyperparameter_search(
        X_train_imp=X_train_imp,
        y_train=y_train,
        X_val_imp=X_val_imp,
        y_val=y_val,
        model_profile=model_profile,
        approach=preprocess_profile.get("approach"),
        sequence_length=T,
        n_features=len(feature_cols),
        verbose=True,
    )

    _, metrics, _ = evaluate_test_set(
        final_model=final_model,
        X_test_imp=X_test_imp,
        y_test=y_test,
        model_profile=model_profile,
        target_profile=target_profile,
        show_plots=False,
    )

    report_info = None
    if bool(output_profile.get("save_visual_report", True)):
        report_info = export_visual_report(
            dataset_df=dataset_df,
            feature_cols=feature_cols,
            model_profile=model_profile,
            output_profile=output_profile,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            target_profile=target_profile,
            raw_df=raw_df,
            preprocess_profile=preprocess_profile,
            best_params=best_params,
            metrics=metrics,
            final_model=final_model,
            X_test_imp=X_test_imp,
            y_test=y_test,
            dataset_df_train=dataset_df,
            feature_cols_train=feature_cols,
        )

    return metrics, report_info


def _run_stacking(name, profiles):
    """Stacking multistream : un modèle par flux → méta-features → méta-modèle.

    Requiert raw_profile (segmentation brute 3D) et fusion_profile avec :
      streams     : {flux_name: [col_name, ...]}
      meta_model  : type du méta-modèle (defaut: logistic_regression)
    """
    data_profile       = profiles["data_profile"]
    preprocess_profile = profiles["preprocess_profile"]
    target_profile     = profiles["target_profile"]
    model_profile      = profiles["model_profile"]
    fusion_profile     = profiles["fusion_profile"]
    raw_profile        = profiles.get("raw_profile")
    output_profile     = profiles["output_profile"]

    if not raw_profile:
        raise ValueError("Le stacking multistream requiert raw_profile (segmentation 3D).")

    # ── 1. Chargement + prétraitement ────────────────────────────────────────
    features_df = load_features_for_approach(
        data_profile=data_profile,
        preprocess_profile=preprocess_profile,
        verbose=True,
    )
    features_df = apply_column_aggregations(features_df, preprocess_profile)

    if raw_profile.get("per_subject_norm", False):
        norm_cols = [
            c for c in preprocess_profile.get("include_features", [])
            if c in features_df.columns and pd.api.types.is_numeric_dtype(features_df[c])
        ]
        if "Pupil diameter AVG" in features_df.columns and "Pupil diameter AVG" not in norm_cols:
            norm_cols.append("Pupil diameter AVG")
        for sid, idx in features_df.groupby("subject_id", sort=False).groups.items():
            sub   = features_df.loc[idx, norm_cols].astype(float)
            mu    = sub.mean()
            sigma = sub.std().replace(0, 1e-8)
            features_df.loc[idx, norm_cols] = (sub - mu) / sigma
        print(f"Normalisation per-sujet : {features_df['subject_id'].nunique()} sujets, {len(norm_cols)} features")

    target_df = load_per_minute_targets(target_profile)
    target_df["subject_id"] = target_df["subject_id"].astype(str).str.strip()
    target_df = apply_target_discretization(target_df, target_profile)
    print("Distribution cibles :")
    for k, v in sorted(collections.Counter(target_df["target"].tolist()).items()):
        print(f"  {k}: {v}")

    proc_df, feature_cols = apply_preprocess(features_df.copy(), preprocess_profile)

    window_s      = float(raw_profile["window_s"])
    stride_s      = float(raw_profile["stride_s"])
    T             = int(raw_profile["sequence_length"])
    session_min_s = raw_profile.get("session_min_s")
    session_max_s = raw_profile.get("session_max_s")
    print(f"Segmentation - fenetre {window_s}s, stride {stride_s}s, T={T} ...")

    X, y, groups = segment_sequences_sliding_window(
        raw_df=proc_df,
        feature_cols=feature_cols,
        target_df=target_df,
        window_s=window_s,
        stride_s=stride_s,
        T=T,
        subject_col="subject_id",
        time_col="time",
        pad_value=float(raw_profile.get("pad_value", 0.0)),
        session_min_s=float(session_min_s) if session_min_s is not None else None,
        session_max_s=float(session_max_s) if session_max_s is not None else None,
    )
    print(f"X shape : {X.shape}  (n_fenetres, T, n_features)")

    model_profile["n_classes"]       = int(len(np.unique(y)))
    model_profile["sequence_length"] = T
    model_profile["n_features"]      = int(len(feature_cols))

    data_source = str(data_profile.get("data_source", "phase1")).strip().lower()
    use_root    = data_source == "both" or bool(model_profile.get("group_split_by_root_subject", False))
    groups_for_split = (
        np.array([re.sub(r"_P\d+$", "", g) for g in groups], dtype=object)
        if use_root else groups
    )

    prepared = prepare_splits_and_impute_3d(
        X=X, y=y, groups=groups_for_split,
        preprocess_profile=preprocess_profile,
        model_profile=model_profile,
    )
    X_train_imp = prepared["X_train_imp"]
    X_val_imp   = prepared["X_val_imp"]
    X_test_imp  = prepared["X_test_imp"]
    y_train     = prepared["y_train"]
    y_val       = prepared["y_val"]
    y_test      = prepared["y_test"]
    train_idx   = prepared["train_idx"]
    val_idx     = prepared["val_idx"]
    test_idx    = prepared["test_idx"]
    print(f"Split train/val/test : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    # ── 2. Stacking ───────────────────────────────────────────────────────────
    is_classif   = str(model_profile.get("task_type", "classification")).lower() == "classification"
    feature_list = list(feature_cols)

    stream_map = split_feature_streams(feature_list, fusion_profile)
    print(f"[stacking] Flux : { {k: len(v) for k, v in stream_map.items()} }")

    stream_models = train_stream_models(
        X_train_imp, y_train, X_val_imp, y_val,
        stream_map, feature_list, model_profile,
    )

    meta_train = make_meta_features(stream_models, stream_map, X_train_imp, feature_list, is_classif)
    meta_test  = make_meta_features(stream_models, stream_map, X_test_imp,  feature_list, is_classif)
    print(f"[stacking] Meta-features shape : {meta_train.shape}")

    n_classes  = int(len(np.unique(y_train)))
    meta_model = build_meta_model(
        fusion_profile=fusion_profile,
        task_type=model_profile.get("task_type", "classification"),
        seed=model_profile.get("random_state", 42),
        n_meta_features=meta_train.shape[1],
        n_classes=n_classes,
    )
    meta_model.fit(meta_train, y_train)

    # ── 3. Évaluation ─────────────────────────────────────────────────────────
    _, metrics, _ = evaluate_test_set(
        final_model=meta_model,
        X_test_imp=meta_test,
        y_test=y_test,
        model_profile=model_profile,
        target_profile=target_profile,
        show_plots=False,
    )

    # ── 4. Rapport visuel ─────────────────────────────────────────────────────
    dataset_df = pd.DataFrame(X.mean(axis=1), columns=feature_cols)
    dataset_df["target"]     = y
    dataset_df["subject_id"] = groups
    raw_df = dataset_df.copy()

    report_info = None
    if bool(output_profile.get("save_visual_report", True)):
        report_info = export_visual_report(
            dataset_df=dataset_df,
            feature_cols=feature_cols,
            model_profile=model_profile,
            output_profile=output_profile,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            target_profile=target_profile,
            raw_df=raw_df,
            preprocess_profile=preprocess_profile,
            best_params={},
            metrics=metrics,
            final_model=meta_model,
            X_test_imp=meta_test,
            y_test=y_test,
            dataset_df_train=dataset_df,
            feature_cols_train=feature_cols,
        )

    return metrics, report_info


def run_hypothesis(hypothesis, base_profiles, repo_root, default_seed=None):
    name = hypothesis.get("name", "hypothesis")
    slug = _slugify(name)

    profiles       = _prepare_profiles(repo_root, base_profiles, hypothesis.get("profiles", {}))
    model_profile  = profiles["model_profile"]
    output_profile = profiles["output_profile"]

    seed = int(hypothesis.get("seed", default_seed if default_seed is not None else 42))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    model_profile["random_state"] = seed

    if "output_dir" in output_profile:
        output_profile["output_dir"] = str(Path(output_profile["output_dir"]) / slug)
    output_profile.setdefault("visual_report_name", slug)

    result = {"name": name, "slug": slug, "status": "success", "seed": seed}

    try:
        print(f"\n=== [{name}] Start ===")
        use_stacking = "fusion_profile" in profiles and profiles["fusion_profile"]
        use_raw      = "raw_profile" in profiles and profiles["raw_profile"]
        if use_stacking:
            print("Mode : stacking multistream (fusion_profile detecte).")
            metrics, report_info = _run_stacking(name, profiles)
        elif use_raw:
            print("Mode : segmentation brute 3D (raw_profile detecte).")
            metrics, report_info = _run_raw(name, profiles)
        else:
            print("Mode : pipeline standard 2D.")
            metrics, report_info = _run_standard(name, profiles)

        result["metrics"] = _jsonify(metrics)
        result["report"]  = _jsonify(report_info)
        print(f"=== [{name}] Done ===")

    except Exception as exc:
        result["status"]    = "failed"
        result["error"]     = str(exc)
        result["traceback"] = traceback.format_exc()
        print(f"=== [{name}] Failed: {exc} ===")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple configurable hypotheses sequentially (overnight batch)."
    )
    parser.add_argument("--config", required=True,
                        help="Path to a JSON config file.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop at first failing hypothesis.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only selected hypothesis names.")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent

    config_path = _resolve_config_path(args.config, script_dir, repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed              = cfg.get("seed", 42)
    continue_on_error = bool(cfg.get("continue_on_error", True))
    base_profiles     = cfg.get("base_profiles", {})
    hypotheses        = cfg.get("hypotheses", [])

    if not hypotheses:
        raise ValueError("No hypothesis configured. Fill 'hypotheses' in your JSON config.")

    if args.only:
        selected   = set(args.only)
        hypotheses = [h for h in hypotheses if h.get("name") in selected]
        if not hypotheses:
            raise ValueError("No hypothesis matched --only selection.")

    print(f"Hypotheses to run: {len(hypotheses)}")

    summary     = []
    has_failure = False
    for hypothesis in hypotheses:
        run_result  = run_hypothesis(
            hypothesis=hypothesis,
            base_profiles=base_profiles,
            repo_root=repo_root,
            default_seed=seed,
        )
        summary.append(run_result)
        failed = run_result.get("status") != "success"
        has_failure = has_failure or failed
        if failed and (args.fail_fast or not continue_on_error):
            print("Stopping batch due to failure.")
            break

    print("\n" + "=" * 60)
    for r in summary:
        status  = r.get("status", "?")
        metrics = r.get("metrics", {})
        acc     = metrics.get("accuracy", metrics.get("test_accuracy", "n/a"))
        f1      = metrics.get("f1_macro", metrics.get("test_f1_macro", "n/a"))
        print(f"  [{status:^7}] {r['name']}  acc={acc}  f1={f1}")
    print("=" * 60)

    if has_failure:
        print("Batch completed with at least one failure.")
        return 1
    print("Batch completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
