import argparse
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
from pipeline.extract import add_target, load_features_for_approach
from pipeline.pretraitement import (
    apply_preprocess,
    apply_target_discretization,
    prepare_splits_and_impute,
)
from pipeline.reporting import export_visual_report


DEEP_MODELS = {"cnn_1d", "inception_time", "bilstm", "cnn_lstm"}


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
    """Résout un chemin ou une liste de chemins.

    Si path_value est une liste, résout chaque chemin et retourne une liste.
    Si path_value est une chaîne, résout et retourne une chaîne (ou Path).
    """
    # Gérer les listes de chemins (multi-source)
    if isinstance(path_value, (list, tuple)):
        resolved = []
        for pv in path_value:
            path = Path(pv)
            if path.is_absolute():
                resolved.append(str(path))
            else:
                resolved.append(str((repo_root / path).resolve()))
        return resolved

    # Cas standard: chaîne unique
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

    # Fallback for clearer error message using caller-provided relative path.
    return candidates[0]


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
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

    if target_profile.get("source") == "csv" and "csv_path" in target_profile:
        resolved = _resolve_path(target_profile["csv_path"], repo_root)
        target_profile["csv_path"] = resolved if isinstance(resolved, list) else str(resolved)
    if target_profile.get("source") == "xlsx" and "xlsx_path" in target_profile:
        resolved = _resolve_path(target_profile["xlsx_path"], repo_root)
        target_profile["xlsx_path"] = resolved if isinstance(resolved, list) else str(resolved)

    if "output_dir" in output_profile:
        output_profile["output_dir"] = str(_resolve_path(output_profile["output_dir"], repo_root))

    return profiles


def run_hypothesis(hypothesis, base_profiles, repo_root, default_seed=None):
    name = hypothesis.get("name", "hypothesis")
    slug = _slugify(name)

    profiles = _prepare_profiles(repo_root, base_profiles, hypothesis.get("profiles", {}))
    data_profile = profiles["data_profile"]
    preprocess_profile = profiles["preprocess_profile"]
    target_profile = profiles["target_profile"]
    model_profile = profiles["model_profile"]
    output_profile = profiles["output_profile"]

    seed = int(hypothesis.get("seed", default_seed if default_seed is not None else 42))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    model_profile["random_state"] = seed

    if "output_dir" in output_profile:
        output_profile["output_dir"] = str(Path(output_profile["output_dir"]) / slug)
    output_profile.setdefault("visual_report_name", slug)

    result = {
        "name": name,
        "slug": slug,
        "status": "success",
        "seed": seed,
    }

    try:
        print(f"\n=== [{name}] Start ===")

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
        else:
            if target_profile.get("discretize") is not None:
                print("Regression mode: target discretization is ignored.")

        raw_df = dataset_df.copy()
        preprocess_profile_visu = dict(preprocess_profile)
        already_windowed_input = any("__t" in str(c) for c in dataset_df.columns)

        if not already_windowed_input:
            preprocess_profile_visu["window_duration_s"] = None
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
            y_val = pd.to_numeric(pd.Series(y_val), errors="coerce").to_numpy(dtype=float)
            y_test = pd.to_numeric(pd.Series(y_test), errors="coerce").to_numpy(dtype=float)
            if np.isnan(y_train).any() or np.isnan(y_val).any() or np.isnan(y_test).any():
                raise ValueError("Regression target contains NaN after numeric conversion.")

        model_type = str(model_profile.get("model_type", "random_forest")).lower()
        if model_type in DEEP_MODELS:
            X_train_imp, X_val_imp, X_test_imp, sequence_length, n_features = _reshape_for_deep_models(
                X_train_imp, X_val_imp, X_test_imp, feature_cols_train
            )
            model_profile["n_classes"] = int(pd.Series(y_train).nunique()) if task_type == "classification" else 1
            model_profile["sequence_length"] = int(sequence_length)
            model_profile["n_features"] = int(n_features)
        else:
            sequence_length = None
            n_features = None

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
                preprocess_profile=preprocess_profile_visu,
                best_params=best_params,
                metrics=metrics,
                final_model=final_model,
                X_test_imp=X_test_imp,
                y_test=y_test,
                dataset_df_train=dataset_df_train,
                feature_cols_train=feature_cols_train,
            )

        result["metrics"] = _jsonify(metrics)
        result["report"] = _jsonify(report_info)
        print(f"=== [{name}] Done ===")
        return result

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(f"=== [{name}] Failed: {exc} ===")
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple configurable hypotheses sequentially (overnight batch)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file (see hypotheses_config.example.json).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first failing hypothesis.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only selected hypothesis names.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    config_path = _resolve_config_path(args.config, script_dir, repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    continue_on_error = bool(cfg.get("continue_on_error", True))
    base_profiles = cfg.get("base_profiles", {})
    hypotheses = cfg.get("hypotheses", [])
    if len(hypotheses) == 0:
        raise ValueError("No hypothesis configured. Fill 'hypotheses' in your JSON config.")

    if args.only:
        selected = set(args.only)
        hypotheses = [h for h in hypotheses if h.get("name") in selected]
        if len(hypotheses) == 0:
            raise ValueError("No hypothesis matched --only selection.")

    print(f"Hypotheses to run: {len(hypotheses)}")

    summary = []
    has_failure = False
    for hypothesis in hypotheses:
        run_result = run_hypothesis(
            hypothesis=hypothesis,
            base_profiles=base_profiles,
            repo_root=repo_root,
            default_seed=seed,
        )
        summary.append(run_result)

        failed = run_result.get("status") != "success"
        has_failure = has_failure or failed
        should_stop = failed and (args.fail_fast or not continue_on_error)
        if should_stop:
            print("Stopping batch due to failure.")
            break

    if has_failure:
        print("Batch completed with at least one failure.")
        return 1

    print("Batch completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
