import json
import os
import math
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pipeline.models import get_search_space
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def build_model_card(
    dataset_df,
    feature_cols,
    model_profile,
    data_profile,
    best_params,
    metrics,
    noise_scores,
    preprocess_profile=None,
):
    dataset_path = data_profile.get("file_path") or data_profile.get("mat_file_path") or "unknown"

    _model_type = model_profile.get("model_type", "random_forest").lower()
    _model_display = "XGBoost" if _model_type == "xgboost" else "RandomForest"

    card = {
        "model_name": _model_display,
        "task_type": model_profile["task_type"],
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

    if preprocess_profile is not None:
        card["normalization"] = preprocess_profile.get("normalization") or "none"
        card["imputation_strategy"] = preprocess_profile.get("imputation_strategy", "median")

    return card


def _compute_ranked_numeric_features(dataset_df, feature_cols):
    numeric_cols = [
        col
        for col in feature_cols
        if col in dataset_df.columns and pd.api.types.is_numeric_dtype(dataset_df[col])
    ]
    if not numeric_cols:
        return []
    variances = dataset_df[numeric_cols].var().sort_values(ascending=False)
    return variances.index.tolist()


def _get_target_order(dataset_df, target_profile):
    configured_order = (target_profile.get("discretize") or {}).get("labels") if target_profile else None
    if configured_order is not None:
        present = set(dataset_df["target"].dropna().unique())
        return [label for label in configured_order if label in present]
    return sorted(dataset_df["target"].dropna().unique(), key=str)


def _build_architecture_blocks(model_type, params, task_type, n_features):
    output_head = "Softmax" if task_type == "classification" else "Linear"

    input_color = "#EAF2FF"
    core_color = "#DCEBFF"
    dense_color = "#E8F5E9"
    output_color = "#FFF1D6"

    if model_type == "random_forest":
        return [
            {"title": "Input", "subtitle": f"{n_features} features", "color": input_color},
            {"title": "RandomForest", "subtitle": f"{int(params.get('n_estimators', 200))} trees", "color": core_color},
            {"title": "Output", "subtitle": output_head, "color": output_color},
        ]

    if model_type == "xgboost":
        return [
            {"title": "Input", "subtitle": f"{n_features} features", "color": input_color},
            {"title": "XGBoost", "subtitle": f"{int(params.get('n_estimators', 100))} estimators", "color": core_color},
            {"title": "Output", "subtitle": output_head, "color": output_color},
        ]

    if model_type == "cnn_1d":
        return [
            {"title": "Input", "subtitle": "Sequence", "color": input_color},
            {"title": "Conv1D", "subtitle": f"f={int(params.get('filters', 32))}, k={int(params.get('kernel_size', 3))}", "color": "#DCEBFF"},
            {"title": "Conv1D", "subtitle": f"f={int(params.get('filters', 32)) * 2}", "color": "#CBE8F6"},
            {"title": "GAP", "subtitle": "GlobalAveragePooling1D", "color": dense_color},
            {"title": "Dense", "subtitle": "64 units", "color": "#DFF3E3"},
            {"title": "Output", "subtitle": output_head, "color": output_color},
        ]

    if model_type == "inception_time":
        return [
            {"title": "Input", "subtitle": "Sequence", "color": input_color},
            {"title": "Inception", "subtitle": "Conv k=1 / 3 / 5", "color": "#DCEBFF"},
            {"title": "Stack", "subtitle": f"depth={int(params.get('depth', 2))}", "color": "#CBE8F6"},
            {"title": "GAP", "subtitle": "GlobalAveragePooling1D", "color": dense_color},
            {"title": "Dense", "subtitle": "64 units", "color": "#DFF3E3"},
            {"title": "Output", "subtitle": output_head, "color": output_color},
        ]

    if model_type == "bilstm":
        return [
            {"title": "Input", "subtitle": "Sequence", "color": input_color},
            {"title": "BiLSTM", "subtitle": f"units={int(params.get('units', 32))}", "color": "#DCEBFF"},
            {"title": "BiLSTM", "subtitle": f"units={int(params.get('units', 32))}", "color": "#CBE8F6"},
            {"title": "Dense", "subtitle": "64 units", "color": "#DFF3E3"},
            {"title": "Output", "subtitle": output_head, "color": output_color},
        ]

    if model_type == "cnn_lstm":
        return [
            {"title": "Input", "subtitle": "Sequence", "color": input_color},
            {"title": "Conv1D", "subtitle": f"f={int(params.get('cnn_filters', 32))}, k={int(params.get('cnn_kernel', 3))}", "color": "#DCEBFF"},
            {"title": "Conv1D", "subtitle": f"f={int(params.get('cnn_filters', 32)) * 2}", "color": "#CBE8F6"},
            {"title": "LSTM", "subtitle": f"units={int(params.get('lstm_units', 32))}", "color": "#D7EEF2"},
            {"title": "Dense", "subtitle": "64 units", "color": "#DFF3E3"},
            {"title": "Output", "subtitle": output_head, "color": output_color},
        ]

    return [
        {"title": "Input", "subtitle": f"{n_features} features", "color": input_color},
        {"title": "Model", "subtitle": model_type or "unknown", "color": core_color},
        {"title": "Output", "subtitle": output_head, "color": output_color},
    ]


def _draw_architecture_schema(ax, blocks):
    ax.set_axis_off()
    n = len(blocks)
    if n == 0:
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    panel = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#D6DFEA",
        facecolor="#FBFCFE",
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(panel)

    y = 0.48
    box_w = min(0.18, 0.78 / max(1, n))
    box_h = 0.22
    start_x = 0.5 - ((n * box_w) + ((n - 1) * 0.04)) / 2

    for i, block in enumerate(blocks):
        x = start_x + i * (box_w + 0.04)
        shadow = FancyBboxPatch(
            (x + 0.006, y - box_h / 2 - 0.006),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=0,
            facecolor="#000000",
            alpha=0.05,
            transform=ax.transAxes,
            zorder=1,
        )
        ax.add_patch(shadow)
        rect = FancyBboxPatch(
            (x, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.4,
            edgecolor="#6D86A6",
            facecolor=block.get("color", "#E9F1FB"),
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.03,
            y + 0.06,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=8,
            color="#445",
            fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="circle,pad=0.18", facecolor="#FFFFFF", edgecolor="#B8C4D4", linewidth=0.8),
            zorder=3,
        )
        ax.text(
            x + box_w / 2,
            y + 0.035,
            block["title"],
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#1F2D3D",
            transform=ax.transAxes,
            zorder=3,
        )
        ax.text(
            x + box_w / 2,
            y - 0.06,
            block["subtitle"],
            ha="center",
            va="center",
            fontsize=8.2,
            color="#304050",
            transform=ax.transAxes,
            zorder=3,
        )

        if i < n - 1:
            x_next = start_x + (i + 1) * (box_w + 0.04)
            arr = FancyArrowPatch(
                (x + box_w, y),
                (x_next, y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.1,
                color="#7E93B2",
                transform=ax.transAxes,
                zorder=4,
            )
            ax.add_patch(arr)


def _infer_default_params(task_type, model_profile):
    search_space = get_search_space(task_type, model_profile=model_profile)
    if not isinstance(search_space, dict):
        return {}

    defaults = {}
    for key, values in search_space.items():
        if isinstance(values, (list, tuple)) and len(values) > 0:
            defaults[key] = values[0]
        else:
            defaults[key] = values
    return defaults


def _summarize_architecture_rows(model_type, task_type, best_params, model_profile=None):
    rows = [
        ["Model type", model_type],
        ["Task type", task_type],
    ]

    if best_params:
        for key in sorted(best_params.keys()):
            rows.append([str(key), str(best_params[key])])
    else:
        default_params = _infer_default_params(task_type, model_profile or {"model_type": model_type})
        if default_params:
            rows.append(["Hyperparameters", "valeurs par defaut du modele"])
            for key in sorted(default_params.keys()):
                rows.append([str(key), str(default_params[key])])
        else:
            rows.append(["Hyperparameters", "valeurs par defaut indisponibles"])

    return rows


def visual_model_architecture_page(context, save_figure):
    model_profile = context["model_profile"] or {}
    feature_cols = context["feature_cols"] or []
    best_params = context.get("best_params") or {}

    model_type = str(model_profile.get("model_type", "random_forest")).lower()
    task_type = str(model_profile.get("task_type", "unknown"))

    blocks = _build_architecture_blocks(
        model_type=model_type,
        params=best_params,
        task_type=task_type,
        n_features=len(feature_cols),
    )
    rows = _summarize_architecture_rows(model_type, task_type, best_params, model_profile=model_profile)

    fig = plt.figure(figsize=(11.7, 8.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])
    ax_schema = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    fig.suptitle("Architecture du modele", fontsize=15, fontweight="bold")
    _draw_architecture_schema(ax_schema, blocks)

    ax_table.axis("off")
    table = ax_table.table(
        cellText=rows,
        colLabels=["Element", "Valeur"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.08, 0.02, 0.84, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eefc")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fc")

    fig.tight_layout()
    save_figure(fig, "model_architecture")


def visual_cover_page(context, save_figure):
    dataset_df = context["dataset_df"]
    feature_cols = context["feature_cols"]
    model_profile = context["model_profile"]
    preprocess_profile = context.get("preprocess_profile") or {}
    target_profile = context.get("target_profile") or {}
    output_profile = context["output_profile"]

    task_type = model_profile.get("task_type", "unknown")
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")

    ax.text(
        0.5,
        0.985,
        "Compte rendu visuel",
        va="top",
        ha="center",
        fontsize=19,
        fontweight="bold",
        transform=ax.transAxes,
    )

    def _has_explicit(profile, key):
        if key not in profile:
            return False
        value = profile.get(key)
        return value is not None

    def _fmt_value(value):
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    summary_rows = [
        ["N samples", str(int(len(dataset_df)))],
        [
            "N subjects",
            str(int(dataset_df["subject_id"].nunique()) if "subject_id" in dataset_df.columns else "n/a"),
        ],
        ["N features", str(int(len(feature_cols)))],
    ]

    model_keys = [
        "model_type",
        "task_type",
        "split_method",
        "test_size",
        "val_size"
    ]
    if str(task_type).lower() == "classification":
        model_keys.append("class_weight")
    for key in model_keys:
        if _has_explicit(model_profile, key):
            summary_rows.append([f"model.{key}", _fmt_value(model_profile.get(key))])

    preprocess_keys = [
        "use_frequency_resampling",
        "frequency_resampling_method",
        "uniform_time_step_s",
        "frequency_sampling_hz",
    ]
    for key in preprocess_keys:
        if _has_explicit(preprocess_profile, key):
            summary_rows.append([f"preprocess.{key}", _fmt_value(preprocess_profile.get(key))])

    approach = str(preprocess_profile.get("approach", "A")).strip().upper()
    if approach == "B":
        summary_rows.append(["preprocess.temporal_filters", "active"])

        for prefix in ("lowpass", "highpass", "bandpass"):
            apply_key = f"apply_temporal_{prefix}"
            enabled = bool(preprocess_profile.get(apply_key, False))
            summary_rows.append([f"preprocess.{apply_key}", str(enabled)])
            if enabled:
                for suffix in ("cutoff_hz", "low_cutoff_hz", "high_cutoff_hz", "order", "min_points", "features"):
                    key = f"{prefix}_{suffix}"
                    if _has_explicit(preprocess_profile, key):
                        summary_rows.append([f"preprocess.{key}", _fmt_value(preprocess_profile.get(key))])

    target_keys = ["target_mode"]
    for key in target_keys:
        if _has_explicit(target_profile, key):
            summary_rows.append([f"target.{key}", _fmt_value(target_profile.get(key))])

    if str(task_type).lower() == "classification":
        discretize = target_profile.get("discretize")
        if isinstance(discretize, dict):
            if discretize.get("bins") is not None:
                summary_rows.append(["target.discretize.bins", _fmt_value(discretize.get("bins"))])
            if discretize.get("labels") is not None:
                summary_rows.append(["target.discretize.labels", _fmt_value(discretize.get("labels"))])

    table_bottom = 0.03
    table_height = 0.90
    table = ax.table(
        cellText=summary_rows,
        colLabels=["Caracteristique", "Valeur"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.01, table_bottom, 0.98, table_height],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.15)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eefc")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fc")

    save_figure(fig, "page_de_garde")


def visual_hypothesis_page(context, save_figure):
    output_profile = context["output_profile"]
    hypothesis = output_profile.get("hypothesis")
    if not hypothesis:
        return

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "Compte rendu visuel",
        va="top",
        ha="center",
        fontsize=20,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.90,
        datetime.now().strftime("%d/%m/%Y %H:%M"),
        va="top",
        ha="center",
        fontsize=11,
        color="#555555",
        transform=ax.transAxes,
    )

    ax.text(
        0.04,
        0.72,
        str(hypothesis).strip(),
        va="top",
        ha="left",
        fontsize=12,
        wrap=True,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff", edgecolor="#aabbdd"),
    )

    save_figure(fig, "hypothesis_page")


def visual_split_report(context, save_figure):
    dataset_df = context["dataset_df"]
    model_profile = context["model_profile"]
    train_idx = context["train_idx"]
    val_idx = context["val_idx"]
    test_idx = context["test_idx"]

    if train_idx is None or val_idx is None or test_idx is None or "target" not in dataset_df.columns:
        return

    task_type = model_profile.get("task_type", "unknown")
    splits = {"Train": train_idx, "Val": val_idx, "Test": test_idx}
    is_classif = task_type == "classification"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    subject_counts = {
        name: dataset_df.iloc[idx]["subject_id"].nunique() if "subject_id" in dataset_df.columns else len(idx)
        for name, idx in splits.items()
    }
    bars = axes[0].bar(subject_counts.keys(), subject_counts.values(), color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Sujets uniques par split")
    axes[0].set_ylabel("Nombre")
    for bar, value in zip(bars, subject_counts.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(value), ha="center", va="bottom")

    dfs = []
    for name, idx in splits.items():
        tmp = dataset_df.iloc[idx][["target"]].copy()
        tmp["split"] = name
        dfs.append(tmp)
    merged = pd.concat(dfs, ignore_index=True)
    merged["split"] = pd.Categorical(merged["split"], categories=["Train", "Val", "Test"], ordered=True)

    if is_classif:
        counts = merged.groupby(["split", "target"], observed=True).size().reset_index(name="n")
        totals = counts.groupby("split")["n"].transform("sum")
        counts["pct"] = counts["n"] / totals * 100
        sns.barplot(data=counts, x="split", y="pct", hue="target", ax=axes[1])
        axes[1].set_title("Distribution des classes par split (%)")
        axes[1].set_ylabel("%")
    else:
        for name, color in zip(["Train", "Val", "Test"], ["#4C72B0", "#DD8452", "#55A868"]):
            subset = merged[merged["split"] == name]["target"]
            axes[1].hist(subset, bins=15, alpha=0.6, label=name, color=color)
        axes[1].legend()
        axes[1].set_title("Distribution de la cible par split")

    fig.suptitle("Repartition du split", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "split_report")


def visual_correlation_pages(context, save_figure):
    dataset_df = context["dataset_df"]
    ranked_num_cols = context["ranked_num_cols"]
    max_corr_features = context["output_profile"].get("max_corr_features", 24)

    if len(ranked_num_cols) < 2:
        return

    n_corr_pages = math.ceil(len(ranked_num_cols) / max_corr_features)
    for i in range(n_corr_pages):
        chunk = ranked_num_cols[i * max_corr_features : (i + 1) * max_corr_features]
        if len(chunk) < 2:
            continue
        corr = dataset_df[chunk].corr()
        size = max(8, len(chunk) * 0.52)
        fig, ax = plt.subplots(figsize=(size, size * 0.8))
        sns.heatmap(
            corr,
            annot=len(chunk) <= 15,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.25,
            annot_kws={"size": 7},
            ax=ax,
        )
        ax.set_title(f"Correlation (page {i + 1}/{n_corr_pages}) - {len(chunk)} features")
        fig.tight_layout()
        save_figure(fig, f"corr_page_{i + 1}")


def visual_violin_pages(context, save_figure):
    dataset_df = context["dataset_df"]
    model_profile = context["model_profile"]
    target_profile = context["target_profile"]
    ranked_num_cols = context["ranked_num_cols"]
    max_violin_features = context["output_profile"].get("max_violin_features", 48)
    violin_features_per_page = context["output_profile"].get("violin_features_per_page", 6)

    task_type = model_profile.get("task_type", "unknown")
    if task_type != "classification" or len(ranked_num_cols) == 0 or "target" not in dataset_df.columns:
        return

    violin_cols = ranked_num_cols[:max_violin_features]
    target_order = _get_target_order(dataset_df, target_profile)
    if not target_order:
        return

    n_v_pages = math.ceil(len(violin_cols) / violin_features_per_page)
    for i in range(n_v_pages):
        chunk = violin_cols[i * violin_features_per_page : (i + 1) * violin_features_per_page]
        if not chunk:
            continue
        n_cols_grid = 2
        n_rows = math.ceil(len(chunk) / n_cols_grid)
        fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(14, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for j, col in enumerate(chunk):
            sns.violinplot(
                data=dataset_df,
                x="target",
                y=col,
                order=target_order,
                ax=axes[j],
                inner="box",
                cut=0,
                palette="Set2",
            )
            axes[j].set_title(col, fontsize=10)
            axes[j].set_xlabel("")

        for j in range(len(chunk), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Distribution des features par classe (page {i + 1}/{n_v_pages})", fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_figure(fig, f"violin_page_{i + 1}")


def visual_missing_values_bar(context, save_figure):
    raw_df = context["raw_df"]
    feature_cols = context["feature_cols"]
    if raw_df is None or len(feature_cols) == 0:
        return

    cols_available = [c for c in feature_cols if c in raw_df.columns]
    if not cols_available:
        return

    nan_pct = raw_df[cols_available].isna().mean() * 100
    nan_pct = nan_pct[nan_pct > 0].sort_values(ascending=False)
    if nan_pct.empty:
        return

    top_nan = nan_pct.head(40)
    fig, ax = plt.subplots(figsize=(max(10, len(top_nan) * 0.25), 4.5))
    sns.barplot(x=top_nan.index, y=top_nan.values, ax=ax)
    ax.set_title("Top features avec valeurs manquantes (%)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("% NaN")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, "missing_values_bar")


def visual_clipping_boxplots(context, save_figure):
    raw_df = context["raw_df"]
    preprocess_profile = context["preprocess_profile"]
    ranked_num_cols = context["ranked_num_cols"]
    if raw_df is None or preprocess_profile is None or preprocess_profile.get("clip_quantiles") is None:
        return

    q_low, q_high = preprocess_profile["clip_quantiles"]
    clip_cols = [c for c in ranked_num_cols[:6] if c in raw_df.columns]
    if not clip_cols:
        return

    clipped_df = raw_df.copy()
    for col in clip_cols:
        clipped_df[col] = clipped_df[col].clip(
            lower=clipped_df[col].quantile(q_low),
            upper=clipped_df[col].quantile(q_high),
        )

    n_cols_grid = 3
    n_rows = math.ceil(len(clip_cols) / n_cols_grid)
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(15, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(clip_cols):
        sns.boxplot(data=[raw_df[col], clipped_df[col]], ax=axes[i])
        axes[i].set_title(f"Feature: {col}")
        axes[i].set_xticklabels(["Avant", "Apres"])
    for i in range(len(clip_cols), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Impact du clipping (echantillon de features)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "clipping_boxplots")


def _temporal_subject_stats(df, time_col):
    rows = []
    for sid, g in df.groupby("subject_id", observed=True):
        t = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
        if t.size < 2:
            rows.append({"subject_id": str(sid), "n_samples": int(t.size), "duration_s": 0.0})
            continue
        rows.append(
            {
                "subject_id": str(sid),
                "n_samples": int(t.size),
                "duration_s": float(t[-1] - t[0]),
            }
        )
    return pd.DataFrame(rows)


def _compute_subject_feature_fft(df, subject_id, time_col, feature_name, max_points=4096):
    """Calcule la FFT non normalisee d'une feature pour un sujet."""
    sub = df[df["subject_id"].astype(str) == str(subject_id)].copy()
    if sub.empty or feature_name not in sub.columns:
        return None, None

    tmp = sub[[time_col, feature_name]].copy()
    tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
    tmp[feature_name] = pd.to_numeric(tmp[feature_name], errors="coerce")
    tmp = tmp.dropna(subset=[time_col, feature_name])
    if len(tmp) < 8:
        return None, None

    tmp = tmp.groupby(time_col, as_index=False)[feature_name].mean().sort_values(time_col)
    t = tmp[time_col].to_numpy(dtype=float)
    y = tmp[feature_name].to_numpy(dtype=float)
    if len(t) < 8:
        return None, None

    dt_candidates = np.diff(t)
    dt_candidates = dt_candidates[dt_candidates > 0]
    if len(dt_candidates) == 0:
        return None, None

    dt = float(np.median(dt_candidates))
    if not np.isfinite(dt) or dt <= 0:
        return None, None

    t_min, t_max = float(t[0]), float(t[-1])
    n_points = int(np.floor((t_max - t_min) / dt)) + 1
    if n_points < 8:
        return None, None

    if n_points > max_points:
        n_points = max_points

    t_grid = np.linspace(t_min, t_max, n_points)
    if n_points > 1:
        dt = float(t_grid[1] - t_grid[0])

    y_interp = np.interp(t_grid, t, y)
    y_interp = y_interp - np.nanmean(y_interp)
    if np.allclose(y_interp, 0):
        return None, None

    window = np.hanning(len(y_interp))
    y_win = y_interp * window

    fft_vals = np.fft.rfft(y_win)
    freqs = np.fft.rfftfreq(len(y_win), d=dt)
    power = (np.abs(fft_vals) ** 2).astype(float)
    if len(power) == 0:
        return None, None

    if len(power) > 1:
        power[0] = 0.0

    valid = np.isfinite(freqs) & np.isfinite(power)
    if valid.sum() < 4:
        return None, None

    return freqs[valid], power[valid]


def _resolve_temporal_filtered_features(feature_cols, preprocess_profile):
    if not feature_cols:
        return []

    profile = preprocess_profile or {}
    available_map = {str(c).strip().lower(): c for c in feature_cols}
    selected = []
    seen = set()

    for prefix in ("lowpass", "highpass", "bandpass"):
        if not bool(profile.get(f"apply_temporal_{prefix}", False)):
            continue

        conf = profile.get(f"{prefix}_features", "all")
        if isinstance(conf, str):
            if conf.strip().lower() != "all":
                continue
            candidates = list(feature_cols)
        elif isinstance(conf, (list, tuple, set)):
            candidates = []
            for raw_name in conf:
                key = str(raw_name).strip().lower()
                if key in available_map:
                    candidates.append(available_map[key])
        else:
            continue

        for col in candidates:
            if col not in seen:
                selected.append(col)
                seen.add(col)

    return selected


def _compute_subject_fft_spectrum(df, subject_id, time_col, feature_cols, max_points=4096):
    """Calcule un spectre de puissance moyen (FFT) pour un sujet et un ensemble de features."""
    spectra = []
    freq_axis = None

    for feature_name in feature_cols:
        freqs, power = _compute_subject_feature_fft(
            df,
            subject_id=subject_id,
            time_col=time_col,
            feature_name=feature_name,
            max_points=max_points,
        )
        if freqs is None or power is None:
            continue

        if freq_axis is None:
            freq_axis = freqs
            spectra.append(power)
        else:
            # Reechantillonne si les axes diffèrent legerement selon la feature.
            if len(freqs) != len(freq_axis) or not np.allclose(freqs, freq_axis):
                power = np.interp(freq_axis, freqs, power, left=np.nan, right=np.nan)
            spectra.append(power)

    if not spectra:
        return None, None

    spec_mat = np.vstack(spectra)
    avg_spec = np.nanmedian(spec_mat, axis=0)
    valid = np.isfinite(freq_axis) & np.isfinite(avg_spec)
    if valid.sum() < 4:
        return None, None

    return freq_axis[valid], avg_spec[valid]


def _temporal_fft_cutoff_markers(preprocess_profile):
    profile = preprocess_profile or {}
    markers = []

    if bool(profile.get("apply_temporal_lowpass", False)):
        cutoff = float(profile.get("lowpass_cutoff_hz", 0.0) or 0.0)
        if cutoff > 0:
            markers.append(("LP", cutoff, "#2A9D8F", "--"))

    if bool(profile.get("apply_temporal_highpass", False)):
        cutoff = float(profile.get("highpass_cutoff_hz", 0.0) or 0.0)
        if cutoff > 0:
            markers.append(("HP", cutoff, "#E76F51", "--"))

    if bool(profile.get("apply_temporal_bandpass", False)):
        low_cut = float(profile.get("bandpass_low_cutoff_hz", 0.0) or 0.0)
        high_cut = float(profile.get("bandpass_high_cutoff_hz", 0.0) or 0.0)
        if low_cut > 0:
            markers.append(("BP low", low_cut, "#8E6C8A", ":"))
        if high_cut > 0:
            markers.append(("BP high", high_cut, "#8E6C8A", ":"))

    return markers


def _resolve_confusion_labels(y_true, y_pred, target_profile=None):
    observed = pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)
    observed_labels = [x for x in observed.dropna().unique().tolist()]

    configured_order = None
    if target_profile is not None:
        configured_order = (target_profile.get("discretize") or {}).get("labels")

    if configured_order is not None:
        labels = [c for c in configured_order if c in observed_labels]
        if len(labels) == 0:
            by_str = {str(obs_label): obs_label for obs_label in observed_labels}
            labels = [by_str[str(c)] for c in configured_order if str(c) in by_str]
    else:
        labels = sorted(observed_labels, key=str)

    if len(labels) == 0:
        labels = sorted(observed_labels, key=str)

    return labels


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric_value(value):
    as_float = _safe_float(value)
    if as_float is not None and np.isfinite(as_float):
        return f"{as_float:.4f}"
    return str(value)


def _compute_fallback_metrics(model_profile, final_model, X_test_imp, y_test):
    if final_model is None or X_test_imp is None or y_test is None:
        return None

    task_type = str((model_profile or {}).get("task_type", "classification")).lower()
    y_pred = final_model.predict(X_test_imp)
    y_pred = np.asarray(y_pred)

    if task_type == "classification":
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = np.ravel(y_pred)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    y_pred = np.ravel(y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_test, y_pred),
    }


def visual_metrics_table_page(context, save_figure):
    """Page tableau des metriques d'evaluation (classification/regression, approche A/B)."""
    model_profile = context.get("model_profile") or {}
    preprocess_profile = context.get("preprocess_profile") or {}
    metrics = context.get("metrics")

    if not isinstance(metrics, dict) or len(metrics) == 0:
        metrics = _compute_fallback_metrics(
            model_profile=model_profile,
            final_model=context.get("final_model"),
            X_test_imp=context.get("X_test_imp"),
            y_test=context.get("y_test"),
        )

    if not isinstance(metrics, dict) or len(metrics) == 0:
        return

    task_type = str(model_profile.get("task_type", "unknown"))

    rows = [["task_type", task_type]]
    for key in sorted(metrics.keys()):
        rows.append([str(key), _format_metric_value(metrics[key])])

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.set_title("Tableau des metriques d'evaluation", fontsize=14, fontweight="bold", pad=14)

    table = ax.table(
        cellText=rows,
        colLabels=["Metrique", "Valeur"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.10, 0.10, 0.80, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eefc")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fc")

    fig.tight_layout()
    save_figure(fig, "evaluation_metrics_table")


def visual_confusion_matrix_page(context, save_figure):
    """Page matrice de confusion pour les tâches de classification."""
    model_profile = context.get("model_profile") or {}
    if model_profile.get("task_type") != "classification":
        return

    final_model = context.get("final_model")
    X_test_imp = context.get("X_test_imp")
    y_test = context.get("y_test")
    target_profile = context.get("target_profile")
    if final_model is None or X_test_imp is None or y_test is None:
        return

    pred_test = final_model.predict(X_test_imp)
    labels = _resolve_confusion_labels(y_test, pred_test, target_profile=target_profile)
    if len(labels) == 0:
        return

    cm = confusion_matrix(y_test, pred_test, labels=labels)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Matrice de confusion - Test")
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    fig.tight_layout()
    save_figure(fig, "confusion_matrix")


def visual_temporal_fft_by_feature_pages(context, save_figure):
    """Trace des FFT par feature (avant/apres) pour un sujet representatif."""
    raw_df = context["raw_df"]
    dataset_df = context["dataset_df"]
    preprocess_profile = context.get("preprocess_profile") or {}
    feature_cols = context.get("feature_cols") or []
    output_profile = context.get("output_profile") or {}

    if raw_df is None or raw_df.empty or dataset_df.empty:
        return
    if "subject_id" not in raw_df.columns or "subject_id" not in dataset_df.columns:
        return

    approach = str(preprocess_profile.get("approach", "A")).strip().upper()
    time_col = preprocess_profile.get("time_col", "time")
    if approach != "B" or time_col not in raw_df.columns or time_col not in dataset_df.columns:
        return

    excluded = {"target", "subject_id", "row_id", "minute", "sampling_hz", time_col}
    signal_cols = [
        c for c in feature_cols
        if c in dataset_df.columns and c not in excluded and pd.api.types.is_numeric_dtype(dataset_df[c])
    ]
    if not signal_cols:
        return

    max_fft_features = int(output_profile.get("max_fft_features", 12))
    fft_features_per_page = int(output_profile.get("fft_features_per_page", 4))
    fft_display_max_hz = float(output_profile.get("fft_display_max_hz", 0.5) or 0.5)
    if fft_display_max_hz <= 0:
        fft_display_max_hz = 0.5
    max_fft_features = max(1, max_fft_features)
    fft_features_per_page = max(1, fft_features_per_page)

    filtered_cols = _resolve_temporal_filtered_features(signal_cols, preprocess_profile)
    selected_features = filtered_cols[:max_fft_features]
    if not selected_features:
        return

    subject_counts = dataset_df.groupby("subject_id", observed=True).size().sort_values(ascending=False)
    if subject_counts.empty:
        return
    top_subject = str(subject_counts.index[0])

    cutoff_markers = _temporal_fft_cutoff_markers(preprocess_profile)
    n_pages = math.ceil(len(selected_features) / fft_features_per_page)

    for i in range(n_pages):
        chunk = selected_features[i * fft_features_per_page : (i + 1) * fft_features_per_page]
        if not chunk:
            continue

        n_cols_grid = 2
        n_rows = math.ceil(len(chunk) / n_cols_grid)
        fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(14, 4.2 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for j, feat in enumerate(chunk):
            ax = axes[j]
            raw_freqs, raw_power = _compute_subject_feature_fft(raw_df, top_subject, time_col, feat)
            proc_freqs, proc_power = _compute_subject_feature_fft(dataset_df, top_subject, time_col, feat)

            if raw_freqs is None and proc_freqs is None:
                ax.text(0.5, 0.5, "Donnees insuffisantes", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(feat)
                ax.set_xlabel("Frequence (Hz)")
                ax.set_ylabel("Puissance (non normalisee)")
                ax.set_xlim(0, fft_display_max_hz)
                continue

            if raw_freqs is not None:
                ax.plot(raw_freqs, raw_power, label="Avant", color="#4C72B0", linewidth=1.8, alpha=0.9)
            if proc_freqs is not None:
                ax.plot(proc_freqs, proc_power, label="Apres", color="#55A868", linewidth=1.8, alpha=0.9)

            max_freq = 0.0
            if raw_freqs is not None and len(raw_freqs) > 0:
                max_freq = max(max_freq, float(np.nanmax(raw_freqs)))
            if proc_freqs is not None and len(proc_freqs) > 0:
                max_freq = max(max_freq, float(np.nanmax(proc_freqs)))

            for marker_name, cutoff_hz, color, style in cutoff_markers:
                if max_freq > 0 and cutoff_hz <= max_freq:
                    ax.axvline(cutoff_hz, color=color, linestyle=style, linewidth=1.1, alpha=0.9, label=marker_name)

            ax.set_title(feat)
            ax.set_xlabel("Frequence (Hz)")
            ax.set_ylabel("Puissance (non normalisee)")
            ax.set_xlim(0, fft_display_max_hz)
            ax.grid(alpha=0.2)

            handles, labels = ax.get_legend_handles_labels()
            dedup = {}
            for h, l in zip(handles, labels):
                if l not in dedup:
                    dedup[l] = h
            ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")

        for j in range(len(chunk), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"FFT par feature filtree - sujet {top_subject} (page {i + 1}/{n_pages})",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        save_figure(fig, f"temporal_fft_by_feature_page_{i + 1}")


def visual_temporal_preprocess_pages(context, save_figure):
    """Pages de diagnostic temporel par feature filtree (dynamique + FFT)."""
    raw_df = context["raw_df"]
    dataset_df = context["dataset_df"]
    preprocess_profile = context["preprocess_profile"] or {}
    feature_cols = context["feature_cols"]
    output_profile = context.get("output_profile") or {}

    if raw_df is None or raw_df.empty or dataset_df.empty:
        return
    if "subject_id" not in raw_df.columns or "subject_id" not in dataset_df.columns:
        return

    time_col = preprocess_profile.get("time_col", "time")
    if time_col not in raw_df.columns or time_col not in dataset_df.columns:
        return

    excluded = {"target", "subject_id", "row_id", "minute", "sampling_hz", time_col}
    signal_cols = [
        c for c in feature_cols
        if c in dataset_df.columns and c not in excluded and pd.api.types.is_numeric_dtype(dataset_df[c])
    ]
    if not signal_cols:
        return

    filtered_features = _resolve_temporal_filtered_features(signal_cols, preprocess_profile)
    if not filtered_features:
        return

    max_features = int(output_profile.get("max_temporal_filtered_features", len(filtered_features)))
    features_per_page = int(output_profile.get("temporal_filtered_features_per_page", 2))
    fft_display_max_hz = float(output_profile.get("fft_display_max_hz", 0.5) or 0.5)
    if fft_display_max_hz <= 0:
        fft_display_max_hz = 0.5
    max_features = max(1, max_features)
    features_per_page = max(1, features_per_page)
    selected_features = filtered_features[:max_features]

    raw_stats = _temporal_subject_stats(raw_df, time_col)
    proc_stats = _temporal_subject_stats(dataset_df, time_col)
    if raw_stats.empty and proc_stats.empty:
        return

    merged = raw_stats.merge(proc_stats, on="subject_id", how="outer", suffixes=("_raw", "_proc")).fillna(0)
    merged = merged.sort_values("n_samples_proc", ascending=False)

    top_subject = merged.iloc[0]["subject_id"] if len(merged) else str(dataset_df["subject_id"].iloc[0])
    cutoff_markers = _temporal_fft_cutoff_markers(preprocess_profile)

    raw_sub = raw_df[raw_df["subject_id"].astype(str) == str(top_subject)].copy().sort_values(time_col)
    proc_sub = dataset_df[dataset_df["subject_id"].astype(str) == str(top_subject)].copy().sort_values(time_col)

    n_pages = math.ceil(len(selected_features) / features_per_page)
    for page_idx in range(n_pages):
        chunk = selected_features[page_idx * features_per_page : (page_idx + 1) * features_per_page]
        if not chunk:
            continue

        fig, axes = plt.subplots(len(chunk), 2, figsize=(15, 4.3 * len(chunk)))
        if len(chunk) == 1:
            axes = np.array([axes])

        for i, feat in enumerate(chunk):
            ax_dyn = axes[i, 0]
            ax_fft = axes[i, 1]

            # Dynamique temporelle avant/apres pour la feature.
            if feat in raw_sub.columns:
                ax_dyn.plot(raw_sub[time_col], raw_sub[feat], alpha=0.35, linewidth=1.1, color="#4C72B0", label="Avant")
            if feat in proc_sub.columns:
                ax_dyn.plot(proc_sub[time_col], proc_sub[feat], alpha=0.95, linewidth=1.8, color="#55A868", label="Apres")
            ax_dyn.set_title(f"Dynamique - {feat}")
            ax_dyn.set_xlabel("Temps (s)")
            ax_dyn.set_ylabel("Valeur")
            ax_dyn.grid(alpha=0.2)
            handles_dyn, labels_dyn = ax_dyn.get_legend_handles_labels()
            dedup_dyn = {}
            for h, l in zip(handles_dyn, labels_dyn):
                if l not in dedup_dyn:
                    dedup_dyn[l] = h
            if dedup_dyn:
                ax_dyn.legend(dedup_dyn.values(), dedup_dyn.keys(), fontsize=8, loc="best")

            # FFT avant/apres pour la meme feature.
            raw_freqs, raw_power = _compute_subject_feature_fft(raw_df, top_subject, time_col, feat)
            proc_freqs, proc_power = _compute_subject_feature_fft(dataset_df, top_subject, time_col, feat)

            if raw_freqs is None and proc_freqs is None:
                ax_fft.text(0.5, 0.5, "Donnees insuffisantes", ha="center", va="center", transform=ax_fft.transAxes)
            else:
                if raw_freqs is not None:
                    ax_fft.plot(raw_freqs, raw_power, label="Avant", color="#4C72B0", linewidth=1.8, alpha=0.9)
                if proc_freqs is not None:
                    ax_fft.plot(proc_freqs, proc_power, label="Apres", color="#55A868", linewidth=1.8, alpha=0.9)

                max_freq = 0.0
                if raw_freqs is not None and len(raw_freqs) > 0:
                    max_freq = max(max_freq, float(np.nanmax(raw_freqs)))
                if proc_freqs is not None and len(proc_freqs) > 0:
                    max_freq = max(max_freq, float(np.nanmax(proc_freqs)))

                for marker_name, cutoff_hz, color, style in cutoff_markers:
                    if max_freq > 0 and cutoff_hz <= max_freq:
                        ax_fft.axvline(cutoff_hz, color=color, linestyle=style, linewidth=1.1, alpha=0.9, label=marker_name)

                handles_fft, labels_fft = ax_fft.get_legend_handles_labels()
                dedup_fft = {}
                for h, l in zip(handles_fft, labels_fft):
                    if l not in dedup_fft:
                        dedup_fft[l] = h
                if dedup_fft:
                    ax_fft.legend(dedup_fft.values(), dedup_fft.keys(), fontsize=8, loc="best")

            ax_fft.set_title(f"FFT - {feat}")
            ax_fft.set_xlabel("Frequence (Hz)")
            ax_fft.set_ylabel("Puissance (non normalisee)")
            ax_fft.set_xlim(0, fft_display_max_hz)
            ax_fft.grid(alpha=0.2)

        fig.suptitle(
            f"Avant/Apres par feature filtree - sujet {top_subject} (page {page_idx + 1}/{n_pages})",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        save_figure(fig, f"temporal_preprocess_filtered_features_page_{page_idx + 1}")


VISUAL_REPORT_FUNCTIONS = {
    "visual_cover_page": visual_cover_page,
    "visual_hypothesis_page": visual_hypothesis_page,
    "visual_model_architecture_page": visual_model_architecture_page,
    "visual_split_report": visual_split_report,
    "visual_metrics_table_page": visual_metrics_table_page,
    "visual_confusion_matrix_page": visual_confusion_matrix_page,
    "visual_correlation_pages": visual_correlation_pages,
    "visual_violin_pages": visual_violin_pages,
    "visual_missing_values_bar": visual_missing_values_bar,
    "visual_clipping_boxplots": visual_clipping_boxplots,
    "visual_temporal_preprocess_pages": visual_temporal_preprocess_pages,
    "visual_temporal_fft_by_feature_pages": visual_temporal_fft_by_feature_pages,
}


DEFAULT_VISUAL_REPORT_FUNCTIONS = list(VISUAL_REPORT_FUNCTIONS.keys())


def _resolve_visual_functions(output_profile):
    selected = output_profile.get("visual_report_functions", "all")
    if isinstance(selected, str):
        if selected.strip().lower() != "all":
            raise ValueError("visual_report_functions doit etre 'all' ou une liste de noms de fonctions.")
        return DEFAULT_VISUAL_REPORT_FUNCTIONS
    if not isinstance(selected, (list, tuple, set)):
        raise ValueError("visual_report_functions doit etre 'all' ou une liste de noms de fonctions.")

    deduped = []
    for name in selected:
        if name not in deduped:
            deduped.append(name)

    unknown = [name for name in deduped if name not in VISUAL_REPORT_FUNCTIONS]
    if unknown:
        available = ", ".join(DEFAULT_VISUAL_REPORT_FUNCTIONS)
        raise ValueError(f"Fonctions visuelles inconnues: {unknown}. Disponibles: {available}")

    return deduped


def export_visual_report(
    dataset_df,
    feature_cols,
    model_profile,
    output_profile,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    target_profile=None,
    raw_df=None,
    preprocess_profile=None,
    best_params=None,
    metrics=None,
    final_model=None,
    X_test_imp=None,
    y_test=None,
):
    """
    Exporte un rapport visuel modulaire compose de visualisations selectionnables.

    Parametres :
    dataset_df                : dataframe apres pretraitement
    feature_cols              : liste des features candidates
    model_profile             : configuration du modele (task_type)
    output_profile            : configuration des sorties (output_dir)
    train_idx                 : indices train (optionnel)
    val_idx                   : indices validation (optionnel)
    test_idx                  : indices test (optionnel)
    target_profile            : configuration cible pour l'ordre des classes (optionnel)
    raw_df                    : dataframe avant pretraitement pour les pages NaN (optionnel)
    preprocess_profile        : configuration pretraitement pour afficher le clipping (optionnel)
    best_params               : meilleurs hyperparametres (optionnel, pour la page architecture)
    metrics                   : dictionnaire de metriques d'evaluation (optionnel, pour le tableau des metriques)
    final_model               : modele final entraine (optionnel, pour la matrice de confusion)
    X_test_imp                : matrice test imputee (optionnel, pour la matrice de confusion)
    y_test                    : cible test (optionnel, pour la matrice de confusion)

    output_profile optionnel :
    visual_report_functions   : "all" ou liste de noms de fonctions de visualisation
                                ex: ["visual_cover_page", "visual_correlation_pages"]
    """
    out_dir = output_profile["output_dir"]
    report_format = output_profile.get("visual_report_format", "pdf")
    report_name = output_profile.get("visual_report_name", "visual_report")
    os.makedirs(out_dir, exist_ok=True)

    fmt = str(report_format).lower().strip()
    if fmt not in {"pdf", "png", "both"}:
        raise ValueError("report_format doit etre 'pdf', 'png' ou 'both'.")

    write_pdf = fmt in {"pdf", "both"}
    write_png = fmt in {"png", "both"}

    _date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    _full_name = f"{report_name}_{_date_suffix}"
    pdf_path = os.path.join(out_dir, f"{_full_name}.pdf") if write_pdf else None
    png_dir = os.path.join(out_dir, f"{_full_name}_png") if write_png else None
    if png_dir is not None:
        os.makedirs(png_dir, exist_ok=True)

    page_count = 0
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    ranked_num_cols = _compute_ranked_numeric_features(dataset_df, feature_cols)
    selected_functions = _resolve_visual_functions(output_profile)

    def _save_figure(fig, title):
        nonlocal page_count
        page_count += 1
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
        if png_dir is not None:
            safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in title.lower()).strip("_")
            fname = f"{page_count:02d}_{safe[:60]}.png" if safe else f"{page_count:02d}_figure.png"
            fig.savefig(os.path.join(png_dir, fname), dpi=180, bbox_inches="tight")
        plt.close(fig)

    context = {
        "dataset_df": dataset_df,
        "feature_cols": feature_cols,
        "model_profile": model_profile,
        "output_profile": output_profile,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "target_profile": target_profile,
        "raw_df": raw_df,
        "preprocess_profile": preprocess_profile,
        "ranked_num_cols": ranked_num_cols,
        "best_params": best_params,
        "metrics": metrics,
        "final_model": final_model,
        "X_test_imp": X_test_imp,
        "y_test": y_test,
    }

    try:
        for function_name in selected_functions:
            VISUAL_REPORT_FUNCTIONS[function_name](context, _save_figure)
    finally:
        if pdf is not None:
            pdf.close()

    return {
        "pages": int(page_count),
        "report_name": _full_name,
        "pdf_path": pdf_path,
        "png_dir": png_dir,
    }
