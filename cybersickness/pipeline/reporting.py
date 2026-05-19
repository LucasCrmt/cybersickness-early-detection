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


def visual_cover_page(context, save_figure):
    dataset_df = context["dataset_df"]
    feature_cols = context["feature_cols"]
    model_profile = context["model_profile"]
    output_profile = context["output_profile"]

    task_type = model_profile.get("task_type", "unknown")
    hypothesis = output_profile.get("hypothesis")
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

    if hypothesis:
        ax.text(
            0.03,
            0.82,
            "Hypothese / Postulat",
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
        ["Split method", str(model_profile.get("split_method", "n/a"))],
        ["Test size", str(model_profile.get("test_size", "n/a"))],
        ["Val size", str(model_profile.get("val_size", "n/a"))],
        ["Seed", str(model_profile.get("random_state", "n/a"))],
        ["N samples", str(int(len(dataset_df)))],
        [
            "N subjects",
            str(int(dataset_df["subject_id"].nunique()) if "subject_id" in dataset_df.columns else "n/a"),
        ],
        ["N features", str(int(len(feature_cols)))],
        ["Top corr/page", str(output_profile.get("max_corr_features", 24))],
        ["Violin total", str(output_profile.get("max_violin_features", 48))],
        ["Violin/page", str(output_profile.get("violin_features_per_page", 6))],
        ["Format export", str(output_profile.get("visual_report_format", "pdf"))],
    ]

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


def visual_temporal_preprocess_pages(context, save_figure):
    """Pages de diagnostic temporel pour l'approche B (series temporelles)."""
    raw_df = context["raw_df"]
    dataset_df = context["dataset_df"]
    preprocess_profile = context["preprocess_profile"] or {}
    feature_cols = context["feature_cols"]

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

    selected_cols = signal_cols[:4]
    raw_stats = _temporal_subject_stats(raw_df, time_col)
    proc_stats = _temporal_subject_stats(dataset_df, time_col)
    if raw_stats.empty and proc_stats.empty:
        return

    merged = raw_stats.merge(proc_stats, on="subject_id", how="outer", suffixes=("_raw", "_proc")).fillna(0)
    merged = merged.sort_values("n_samples_proc", ascending=False).head(10)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(len(merged))

    axes[0, 0].bar(x - 0.2, merged["n_samples_raw"], width=0.4, label="Avant", color="#4C72B0")
    axes[0, 0].bar(x + 0.2, merged["n_samples_proc"], width=0.4, label="Apres", color="#55A868")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(merged["subject_id"], rotation=45, ha="right")
    axes[0, 0].set_title("Densite d'echantillonnage par sujet")
    axes[0, 0].set_ylabel("Nombre d'echantillons")
    axes[0, 0].legend()

    axes[0, 1].bar(x - 0.2, merged["duration_s_raw"], width=0.4, label="Avant", color="#4C72B0")
    axes[0, 1].bar(x + 0.2, merged["duration_s_proc"], width=0.4, label="Apres", color="#55A868")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(merged["subject_id"], rotation=45, ha="right")
    axes[0, 1].set_title("Couverture temporelle par sujet")
    axes[0, 1].set_ylabel("Duree (s)")
    axes[0, 1].legend()

    raw_dt = raw_df.groupby("subject_id", observed=True)[time_col].apply(
        lambda s: pd.Series(np.diff(np.sort(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float))))
    )
    proc_dt = dataset_df.groupby("subject_id", observed=True)[time_col].apply(
        lambda s: pd.Series(np.diff(np.sort(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float))))
    )
    raw_dt = pd.to_numeric(raw_dt, errors="coerce")
    proc_dt = pd.to_numeric(proc_dt, errors="coerce")
    raw_dt = raw_dt[(raw_dt > 0) & np.isfinite(raw_dt)]
    proc_dt = proc_dt[(proc_dt > 0) & np.isfinite(proc_dt)]

    if len(raw_dt) > 0:
        sns.kdeplot(raw_dt, ax=axes[1, 0], label="Avant", color="#4C72B0", fill=True, alpha=0.2)
    if len(proc_dt) > 0:
        sns.kdeplot(proc_dt, ax=axes[1, 0], label="Apres", color="#55A868", fill=True, alpha=0.2)
    axes[1, 0].set_title("Distribution des pas temporels (delta t)")
    axes[1, 0].set_xlabel("Delta t (s)")
    axes[1, 0].set_ylabel("Densite")
    axes[1, 0].legend()

    top_subject = merged.iloc[0]["subject_id"] if len(merged) else str(dataset_df["subject_id"].iloc[0])
    raw_sub = raw_df[raw_df["subject_id"].astype(str) == str(top_subject)].copy().sort_values(time_col)
    proc_sub = dataset_df[dataset_df["subject_id"].astype(str) == str(top_subject)].copy().sort_values(time_col)

    for feat in selected_cols:
        if feat in raw_sub.columns:
            axes[1, 1].plot(raw_sub[time_col], raw_sub[feat], alpha=0.25, linewidth=1.0, label=f"{feat} (avant)")
        if feat in proc_sub.columns:
            axes[1, 1].plot(proc_sub[time_col], proc_sub[feat], alpha=0.9, linewidth=2.0, label=f"{feat} (apres)")

    axes[1, 1].set_title(f"Dynamique de signaux - sujet {top_subject}")
    axes[1, 1].set_xlabel("Temps (s)")
    axes[1, 1].set_ylabel("Valeur")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if len(handles) > 8:
        handles = handles[:8]
        labels = labels[:8]
    axes[1, 1].legend(handles, labels, loc="best", fontsize=8)

    fig.suptitle("Rapport de pretraitement temporel - Approche B", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "temporal_preprocess_overview")


VISUAL_REPORT_FUNCTIONS = {
    "visual_cover_page": visual_cover_page,
    "visual_split_report": visual_split_report,
    "visual_correlation_pages": visual_correlation_pages,
    "visual_violin_pages": visual_violin_pages,
    "visual_missing_values_bar": visual_missing_values_bar,
    "visual_clipping_boxplots": visual_clipping_boxplots,
    "visual_temporal_preprocess_pages": visual_temporal_preprocess_pages,
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
