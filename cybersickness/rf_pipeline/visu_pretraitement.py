import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_split_report(dataset_df, train_idx, val_idx, test_idx, model_profile):
    """
    Visualise la répartition train/val/test :
    - nombre de sujets uniques par split
    - distribution des classes par split (classification) ou histogramme cible (régression)
    """
    splits = {"Train": train_idx, "Val": val_idx, "Test": test_idx}
    is_classif = model_profile["task_type"] == "classification"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Sujets par split ---
    ax = axes[0]
    subject_counts = {
        name: dataset_df.iloc[idx]["subject_id"].nunique()
        for name, idx in splits.items()
    }
    bars = ax.bar(subject_counts.keys(), subject_counts.values(), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Sujets uniques par split")
    ax.set_ylabel("Nombre de sujets")
    for bar, v in zip(bars, subject_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(v),
                ha="center", va="bottom", fontweight="bold")

    # --- Distribution cible par split ---
    ax = axes[1]
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
        sns.barplot(data=counts, x="split", y="pct", hue="target", ax=ax)
        ax.set_title("Distribution des classes par split (%)")
        ax.set_ylabel("% d'observations")
        ax.set_xlabel("")
        ax.legend(title="Classe")
    else:
        for name, color in zip(["Train", "Val", "Test"], ["#4C72B0", "#DD8452", "#55A868"]):
            subset = merged[merged["split"] == name]["target"]
            ax.hist(subset, bins=15, alpha=0.6, label=name, color=color)
        ax.set_title("Distribution de la cible par split")
        ax.set_ylabel("Effectif")
        ax.set_xlabel("Cible")
        ax.legend()

    plt.suptitle("Répartition du split", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_feature_report(dataset_df, feature_cols, model_profile, top_n_corr=20, top_n_box=8):
    """
    Visualisations de l'espace de features :
    - matrice de corrélation (top features par variance)
    - violin plots des features par classe (classification uniquement)
    """
    is_classif = model_profile["task_type"] == "classification"

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(dataset_df[c])]

    # --- Matrice de corrélation ---
    cols_corr = num_cols
    if len(cols_corr) > top_n_corr:
        variances = dataset_df[cols_corr].var()
        cols_corr = variances.nlargest(top_n_corr).index.tolist()

    if len(cols_corr) >= 2:
        corr = dataset_df[cols_corr].corr()
        size = max(8, len(cols_corr) * 0.55)
        plt.figure(figsize=(size, size * 0.8))
        sns.heatmap(
            corr, annot=len(cols_corr) <= 15, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.3, annot_kws={"size": 7},
        )
        plt.title(f"Matrice de corrélation — top {len(cols_corr)} features (par variance)", fontsize=12)
        plt.tight_layout()
        plt.show()

    # --- Violin plots par classe (classification uniquement) ---
    if not is_classif:
        return

    cols_box = num_cols[:top_n_box] if len(num_cols) > top_n_box else num_cols
    if len(cols_box) == 0:
        return

    n_cols_grid = 2
    n_rows = (len(cols_box) + 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    target_order = sorted(dataset_df["target"].dropna().unique(), key=str)
    for i, col in enumerate(cols_box):
        sns.violinplot(
            data=dataset_df, x="target", y=col, order=target_order,
            ax=axes[i], inner="box", cut=0, palette="Set2",
        )
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")

    for j in range(len(cols_box), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Distribution des features par classe — top {len(cols_box)} par variance",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_preprocessing_report(raw_df, processed_df, feature_cols, preprocess_profile):
    """
    Produit un rapport visuel du prétraitement pour l'approche A (imputation + clipping).

    Paramètres :
    raw_df            : DataFrame AVANT apply_preprocess (mais APRÈS add_target)
    processed_df      : DataFrame APRÈS apply_preprocess
    feature_cols      : liste de colonnes features retournée par apply_preprocess
    preprocess_profile: dict de config (pour afficher les seuils utilisés)
    """

    # Stats résumées
    raw_shape = raw_df.shape
    processed_shape = processed_df.shape
    raw_subjects = raw_df["subject_id"].nunique()
    processed_subjects = processed_df["subject_id"].nunique()
    raw_feature_cols = [col for col in raw_df.columns if col not in ["subject_id", "row_id", "target"] and pd.api.types.is_numeric_dtype(raw_df[col])]
    dropped_features = len(raw_feature_cols) - len(feature_cols)
    total_nans = raw_df[raw_feature_cols].isna().sum().sum()
    clip_quantiles = preprocess_profile.get("clip_quantiles")
    print(f"Shape avant prétraitement : {raw_shape}")
    print(f"Shape après prétraitement : {processed_shape}")
    print(f"Nombre de sujets exclus : {raw_subjects - processed_subjects}")
    print(f"Nombre de features droppées (zero-variance) : {dropped_features}")
    print(f"Nombre total de NaN avant imputation : {total_nans}")
    print(f"Seuils de clipping utilisés : {clip_quantiles}")


    # Distribution cible
    plt.figure(figsize=(8, 4))
    sns.histplot(processed_df["target"], kde=True, bins=30)
    plt.title("Distribution de la cible (FMS)")
    plt.xlabel("FMS")
    plt.ylabel("Densité")
    plt.tight_layout()
    plt.show()

    # Barplot % NaN par feature
    nan_pct = raw_df[feature_cols].isna().mean() * 100
    nan_pct = nan_pct[nan_pct > 0].sort_values(ascending=False)
    if not nan_pct.empty:
        plt.figure(figsize=(max(8, len(nan_pct) * 0.5), 4))
        sns.barplot(x=nan_pct.index, y=nan_pct.values)
        plt.title("% de valeurs manquantes par feature")
        plt.xlabel("Feature")
        plt.ylabel("% NaN")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune valeur manquante : barplot ignoré.")

    # Heatmap des valeurs manquantes
    if raw_df[feature_cols].isna().any().any():
        plt.figure(figsize=(12, 6))
        sns.heatmap(raw_df[feature_cols].isna(), cbar=False)
        plt.title("Heatmap des valeurs manquantes (True = NaN)")
        plt.xlabel("Features")
        plt.ylabel("Observations")
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune valeur manquante dans les features avant imputation.")

    # clip quantiles boxplots    
    if clip_quantiles is None:
        print("Bloc 4 : aucun clipping configuré, visualisation ignorée.")
        return

    q_low, q_high = clip_quantiles
    clipped_df = raw_df.copy()
    for col in feature_cols:
        clipped_df[col] = clipped_df[col].clip(
            lower=clipped_df[col].quantile(q_low),
            upper=clipped_df[col].quantile(q_high),
        )

    boxplot_features = feature_cols[:6]
    n_cols = 3
    n_rows = (len(boxplot_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(boxplot_features):
        sns.boxplot(data=[raw_df[col], clipped_df[col]], ax=axes[i])
        axes[i].set_title(f"Feature: {col}")
        axes[i].set_xticklabels(["Avant", "Après"])
    for j in range(len(boxplot_features), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()
