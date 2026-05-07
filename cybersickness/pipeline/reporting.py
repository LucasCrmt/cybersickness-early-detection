import json
import os
import math
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from pipeline.visu_pretraitement import (
    _draw_clipping_boxplots,
    _draw_corr_heatmap,
    _draw_missing_bar,
    _draw_split_report,
    _draw_violin_page,
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


def _humanize_target_mode(target_profile):
    mapping = {
        "fixed_minute": "Minute fixe",
        "mean_all_minutes": "Moyenne sur toutes les minutes",
        "mean_range": "Moyenne sur un intervalle de minutes",
        "last_minute": "Derniere minute de l'expérience",
        "per_minute": "Indicateur malaise par minute",
    }
    if not target_profile:
        return "Non precise"
    target_mode = target_profile.get("target_mode")
    if target_mode is None:
        return "Non precise"
    key = str(target_mode).strip().lower()
    return mapping.get(key, str(target_mode).replace("_", " "))


def _format_class_ranges(target_profile):
    if not target_profile:
        return None
    discretize = target_profile.get("discretize")
    if not discretize:
        return None

    bins = discretize.get("bins")
    labels = discretize.get("labels")
    if not bins or not labels or len(bins) != len(labels) + 1:
        return None

    ranges = []
    for i, label in enumerate(labels):
        lo = bins[i]
        hi = bins[i + 1]
        left_bracket = "[" if i == 0 else "("
        ranges.append(f"{label}: {left_bracket}{lo}, {hi}]")
    return " | ".join(ranges)


def visual_cover_page(context, save_figure):
    dataset_df = context["dataset_df"]
    feature_cols = context["feature_cols"]
    model_profile = context["model_profile"]
    output_profile = context["output_profile"]
    data_profile = context.get("data_profile")
    target_profile = context.get("target_profile")

    task_type = model_profile.get("task_type", "unknown")
    hypothesis = output_profile.get("hypothesis")
    dataset_path = None
    if data_profile is not None:
        dataset_path = data_profile.get("file_path") or data_profile.get("mat_file_path")
    source_file = os.path.basename(str(dataset_path)) if dataset_path else "n/a"
    target_mode = _humanize_target_mode(target_profile if target_profile else None)
    class_ranges = _format_class_ranges(target_profile) if task_type == "classification" else None

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

    if hypothesis:
        ax.text(
            0.03,
            0.82,
            "Hypothèse",
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.03,
            0.76,
            str(hypothesis),
            va="top",
            ha="left",
            fontsize=11,
            wrap=True,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="#aabbdd"),
        )
        info_y = 0.52
    else:
        info_y = 0.78

    summary_rows = [
        ["Modele", str(model_profile.get("model_type", "n/a"))],
        ["Tache", str(task_type)],
        ["Fichier source", source_file],
        ["Mode cible", target_mode],
        ["Split method", str(model_profile.get("split_method", "n/a"))],
        ["Test size", str(model_profile.get("test_size", "n/a"))],
        ["Val size", str(model_profile.get("val_size", "n/a"))],
        ["N samples", str(int(len(dataset_df)))],
        [
            "N subjects",
            str(int(dataset_df["subject_id"].nunique()) if "subject_id" in dataset_df.columns else "n/a"),
        ],
        ["N features", str(int(len(feature_cols)))],
    ]

    if class_ranges is not None:
        summary_rows.append(["Plages de classes", class_ranges])

    table_bottom = 0.08
    table_height = max(0.30, info_y - table_bottom - 0.03)
    table = ax.table(
        cellText=summary_rows,
        colLabels=["Caracteristique", "Valeur"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.03, table_bottom, 0.94, table_height],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.15)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eefc")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fc")

    save_figure(fig, "page_de_garde")


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
    _draw_split_report(axes, dataset_df, splits, is_classif)
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
        _draw_corr_heatmap(ax, corr, len(chunk))
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
        _draw_violin_page(axes, dataset_df, chunk, target_order)

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
    _draw_missing_bar(ax, top_nan)
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

    n_cols_grid = 3
    n_rows = math.ceil(len(clip_cols) / n_cols_grid)
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(15, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    _draw_clipping_boxplots(axes, raw_df, clip_cols, q_low, q_high)
    fig.suptitle("Impact du clipping (echantillon de features)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "clipping_boxplots")


def visual_confusion_matrix(context, save_figure):
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    pred_test = context.get("pred_test")
    y_test = context.get("y_test")
    model_profile = context["model_profile"]
    target_profile = context.get("target_profile")

    if model_profile.get("task_type") != "classification" or pred_test is None or y_test is None:
        return

    observed = set(y_test) | set(pred_test)
    configured_order = (target_profile.get("discretize") or {}).get("labels") if target_profile else None
    labels = [c for c in configured_order if c in observed] if configured_order else sorted(observed, key=str)

    cm = sk_confusion_matrix(y_test, pred_test, labels=labels)
    size = max(5, len(labels) * 1.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Matrice de confusion - ensemble test")
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    fig.tight_layout()
    save_figure(fig, "confusion_matrix")


def visual_metrics_bar(context, save_figure):
    metrics = context.get("metrics")
    model_profile = context["model_profile"]

    if not metrics:
        return

    metric_items = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not metric_items:
        return

    names = list(metric_items.keys())
    values = list(metric_items.values())

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    bars = ax.bar(names, values, color="#4C72B0")
    top = max(values) if values else 1
    ax.set_ylim(0, top * 1.18)
    ax.set_title("Metriques de performance — ensemble test")
    ax.set_ylabel("Score")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_figure(fig, "metrics_bar")


def visual_feature_importance(context, save_figure):
    final_model = context.get("final_model")
    feature_cols = context["feature_cols"]
    model_profile = context["model_profile"]
    top_n = context["output_profile"].get("top_n_importance", 20)

    if final_model is None or not hasattr(final_model, "feature_importances_"):
        return

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    _mt = model_profile.get("model_type", "random_forest").lower()
    title = "XGBoost" if _mt == "xgboost" else "RandomForest"

    fig, ax = plt.subplots(figsize=(10, max(5, len(imp_df) * 0.38)))
    sns.barplot(data=imp_df, x="importance", y="feature", orient="h", ax=ax)
    ax.set_title(f"Importance des features - top {len(imp_df)} ({title})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "feature_importance")


def visual_pca(context, save_figure):
    from sklearn.decomposition import PCA

    X_test_imp = context.get("X_test_imp")
    y_test = context.get("y_test")
    model_profile = context["model_profile"]
    target_profile = context.get("target_profile")

    if model_profile.get("task_type") != "classification" or X_test_imp is None or y_test is None:
        return
    if X_test_imp.shape[1] < 2:
        return

    pca = PCA(n_components=2, random_state=model_profile.get("random_state", 42))
    X_2d = pca.fit_transform(X_test_imp)
    var_ratio = pca.explained_variance_ratio_

    configured_order = (target_profile.get("discretize") or {}).get("labels") if target_profile else None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1], hue=y_test,
        hue_order=configured_order, palette="tab10", s=40, ax=ax,
    )
    ax.set_title("Projection PCA — ensemble test")
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance expliquee)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance expliquee)")
    ax.legend(title="Classe")
    fig.tight_layout()
    save_figure(fig, "pca_projection")


VISUAL_REPORT_FUNCTIONS = {
    "visual_cover_page": visual_cover_page,
    "visual_split_report": visual_split_report,
    "visual_correlation_pages": visual_correlation_pages,
    "visual_violin_pages": visual_violin_pages,
    "visual_missing_values_bar": visual_missing_values_bar,
    "visual_clipping_boxplots": visual_clipping_boxplots,
    "visual_confusion_matrix": visual_confusion_matrix,
    "visual_metrics_bar": visual_metrics_bar,
    "visual_feature_importance": visual_feature_importance,
    "visual_pca": visual_pca,
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
    data_profile=None,
    raw_df=None,
    preprocess_profile=None,
    final_model=None,
    pred_test=None,
    y_test=None,
    metrics=None,
    X_test_imp=None,
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
    data_profile              : configuration des donnees (pour afficher le nom de fichier source) (optionnel)
    raw_df                    : dataframe avant pretraitement pour les pages NaN (optionnel)
    preprocess_profile        : configuration pretraitement pour afficher le clipping (optionnel)
    final_model               : modele entraine (pour importance des features) (optionnel)
    pred_test                 : predictions sur le test set (pour matrice de confusion) (optionnel)
    y_test                    : vraies etiquettes test (optionnel)
    metrics                   : dict de metriques (pour visual_metrics_bar) (optionnel)
    X_test_imp                : features test imputed/scaled (pour PCA) (optionnel)

    output_profile optionnel :
    visual_report_functions   : "all" ou liste de noms de fonctions de visualisation
                                ex: ["visual_cover_page", "visual_correlation_pages"]
    top_n_importance          : nombre de features a afficher dans l'importance (defaut 20)
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
        "data_profile": data_profile,
        "raw_df": raw_df,
        "preprocess_profile": preprocess_profile,
        "ranked_num_cols": ranked_num_cols,
        "final_model": final_model,
        "pred_test": pred_test,
        "y_test": y_test,
        "metrics": metrics,
        "X_test_imp": X_test_imp,
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
