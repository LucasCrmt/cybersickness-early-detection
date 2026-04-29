import json
import os

import numpy as np


def build_model_card(
    dataset_df,
    feature_cols,
    model_profile,
    representation_profile,
    data_profile,
    best_params,
    metrics,
    noise_scores,
):
    dataset_path = data_profile.get("file_path") or data_profile.get("mat_file_path") or "unknown"

    return {
        "model_name": "XGBoost",
        "task_type": model_profile["task_type"],
        "approach": representation_profile["approach"],
        "data_source": data_profile.get("source", "unknown"),
        "dataset": dataset_path,
        "mat_variable": data_profile.get("mat_variable"),
        "n_samples": int(len(dataset_df)),
        "n_subjects": int(dataset_df["subject_id"].nunique()),
        "n_features": int(len(feature_cols)),
        "split_method": model_profile["split_method"],
        "test_size": model_profile["test_size"],
        "val_size": model_profile["val_size"],
        "seed": model_profile["random_state"],
        "best_params": best_params,
        "metrics": metrics,
        "robustness_mean": float(np.mean(noise_scores)) if len(noise_scores) > 0 else None,
        "robustness_std": float(np.std(noise_scores)) if len(noise_scores) > 0 else None,
    }


def save_outputs(model_card, results_df, output_profile):
    out_dir = output_profile["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    card_json = os.path.join(out_dir, "model_card_xgboost.json")
    with open(card_json, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    results_csv = os.path.join(out_dir, "hyperparam_search_results.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8")

    return card_json, results_csv
