import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
