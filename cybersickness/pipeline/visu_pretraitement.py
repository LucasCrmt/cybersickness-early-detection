import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


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


def plot_feature_report(dataset_df, feature_cols, model_profile, top_n_corr=20, top_n_box=8, target_profile=None):
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

    _configured_order = (target_profile.get("discretize") or {}).get("labels") if target_profile else None
    if _configured_order is not None:
        _present = set(dataset_df["target"].dropna().unique())
        target_order = [c for c in _configured_order if c in _present]
    else:
        target_order = sorted(dataset_df["target"].dropna().unique(), key=str)
    for i, col in enumerate(cols_box):
        plot_df = dataset_df[["target", col]].copy()
        y_col = col
        title = col
        if str(col).strip().lower() == "isboat":
            # Affiche isBoat en pourcentage pour rendre la distribution plus lisible.
            values = pd.to_numeric(plot_df[col], errors="coerce")
            y_col = "__isboat_pct__"
            if not values.dropna().empty and ((values.dropna() >= 0) & (values.dropna() <= 1)).all():
                plot_df[y_col] = values * 100.0
            else:
                plot_df[y_col] = values
            title = "isBoat (%)"

        sns.violinplot(
            data=plot_df, x="target", y=y_col, order=target_order,
            ax=axes[i], inner="box", cut=0, palette="Set2",
        )
        axes[i].set_title(title, fontsize=10)
        axes[i].set_xlabel("")
        if str(col).strip().lower() == "isboat":
            axes[i].set_ylabel("Pourcentage (%)")

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

    # Colonnes aggregees n'existent pas dans raw_df -> les exclure des blocs qui operent sur raw_df
    raw_only_feature_cols = [c for c in feature_cols if c in raw_df.columns]

    # Barplot % NaN par feature
    nan_pct = raw_df[raw_only_feature_cols].isna().mean() * 100
    nan_pct = nan_pct[nan_pct > 0].sort_values(ascending=False)
    if not nan_pct.empty:
        plt.figure(figsize=(max(8, len(nan_pct) * 0.5), 4))
        sns.barplot(x=nan_pct.index, y=nan_pct.values)
        plt.title("% de valeurs manquantes par feature (hors colonnes aggregees)")
        plt.xlabel("Feature")
        plt.ylabel("% NaN")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune valeur manquante : barplot ignore.")

    # Heatmap des valeurs manquantes
    if raw_df[raw_only_feature_cols].isna().any().any():
        plt.figure(figsize=(12, 6))
        sns.heatmap(raw_df[raw_only_feature_cols].isna(), cbar=False)
        plt.title("Heatmap des valeurs manquantes (True = NaN)")
        plt.xlabel("Features")
        plt.ylabel("Observations")
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune valeur manquante dans les features avant imputation.")

    # clip quantiles boxplots
    if clip_quantiles is None:
        print("Bloc 4 : aucun clipping configure, visualisation ignoree.")
        return

    q_low, q_high = clip_quantiles
    clipped_df = raw_df.copy()
    for col in raw_only_feature_cols:
        clipped_df[col] = clipped_df[col].clip(
            lower=clipped_df[col].quantile(q_low),
            upper=clipped_df[col].quantile(q_high),
        )

    boxplot_features = raw_only_feature_cols[:6]
    n_cols = 3
    n_rows = (len(boxplot_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(boxplot_features):
        sns.boxplot(data=[raw_df[col], clipped_df[col]], ax=axes[i])
        axes[i].set_title(f"Feature: {col}")
        axes[i].set_xticklabels(["Avant", "Apres"])
    for j in range(len(boxplot_features), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def _collect_temporal_feature_columns(df, feature_cols, time_col):
    excluded = {"target", "subject_id", "row_id", "minute", "sampling_hz", time_col}
    return [
        c for c in feature_cols
        if c in df.columns and c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def _resolve_temporal_filtered_features(feature_cols, preprocess_profile):
    if not feature_cols:
        return []

    available_map = {str(c).strip().lower(): c for c in feature_cols}
    selected = []
    seen = set()

    for prefix in ("lowpass", "highpass", "bandpass"):
        if not bool(preprocess_profile.get(f"apply_temporal_{prefix}", False)):
            continue

        conf = preprocess_profile.get(f"{prefix}_features", "all")
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


def _temporal_subject_stats(df, time_col):
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "n_samples", "duration_s", "median_dt_s"])

    rows = []
    for sid, g in df.groupby("subject_id", observed=True):
        t = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
        if t.size < 2:
            rows.append({"subject_id": str(sid), "n_samples": int(t.size), "duration_s": 0.0, "median_dt_s": np.nan})
            continue
        dt = np.diff(t)
        dt = dt[dt > 0]
        rows.append(
            {
                "subject_id": str(sid),
                "n_samples": int(t.size),
                "duration_s": float(t[-1] - t[0]),
                "median_dt_s": float(np.median(dt)) if dt.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _compute_subject_fft_spectrum(df, subject_id, time_col, feature_cols, max_points=4096):
    """Calcule un spectre de puissance moyen (FFT) non normalise pour un sujet et un ensemble de features."""
    sub = df[df["subject_id"].astype(str) == str(subject_id)].copy()
    if sub.empty:
        return None, None

    sub[time_col] = pd.to_numeric(sub[time_col], errors="coerce")
    sub = sub.dropna(subset=[time_col]).sort_values(time_col)
    if len(sub) < 8:
        return None, None

    spectra = []
    freq_axis = None

    for feat in feature_cols:
        if feat not in sub.columns:
            continue

        tmp = sub[[time_col, feat]].copy()
        tmp[feat] = pd.to_numeric(tmp[feat], errors="coerce")
        tmp = tmp.dropna(subset=[time_col, feat])
        if len(tmp) < 8:
            continue

        # Agrège les temps dupliqués pour garantir une interpolation stable.
        tmp = tmp.groupby(time_col, as_index=False)[feat].mean().sort_values(time_col)
        t = tmp[time_col].to_numpy(dtype=float)
        y = tmp[feat].to_numpy(dtype=float)
        if len(t) < 8:
            continue

        dt_candidates = np.diff(t)
        dt_candidates = dt_candidates[dt_candidates > 0]
        if len(dt_candidates) == 0:
            continue

        dt = float(np.median(dt_candidates))
        if not np.isfinite(dt) or dt <= 0:
            continue

        t_min, t_max = float(t[0]), float(t[-1])
        n_points = int(np.floor((t_max - t_min) / dt)) + 1
        if n_points < 8:
            continue

        if n_points > max_points:
            n_points = max_points
        t_grid = np.linspace(t_min, t_max, n_points)
        if n_points > 1:
            dt = float(t_grid[1] - t_grid[0])

        y_interp = np.interp(t_grid, t, y)
        y_interp = y_interp - np.nanmean(y_interp)
        if np.allclose(y_interp, 0):
            continue

        window = np.hanning(len(y_interp))
        y_win = y_interp * window

        fft_vals = np.fft.rfft(y_win)
        freqs = np.fft.rfftfreq(len(y_win), d=dt)
        power = (np.abs(fft_vals) ** 2).astype(float)
        if len(power) == 0:
            continue

        # Ignore la composante continue pour comparer la dynamique frequentielle.
        if len(power) > 1:
            power[0] = 0.0

        if freq_axis is None:
            freq_axis = freqs
            spectra.append(power)
        else:
            # Rééchantillonne si les axes diffèrent légèrement selon la feature.
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


def plot_temporal_preprocessing_report(
    raw_df,
    processed_df,
    feature_cols,
    preprocess_profile,
    max_subjects=10,
    max_features=4,
):
    """
    Rapport visuel de pretraitement pour l'approche B (series temporelles).

    Cette visualisation est volontairement differente de l'approche A:
    - couverture temporelle et densite d'echantillonnage par sujet
    - evolution des deltas de temps avant/apres pretraitement
    - comparaison de la dynamique d'un sous-ensemble de signaux
    """
    time_col = preprocess_profile.get("time_col", "time")
    if time_col not in raw_df.columns or time_col not in processed_df.columns:
        raise ValueError(
            f"Colonne temporelle '{time_col}' absente. Colonnes raw: {list(raw_df.columns)}"
        )

    if "subject_id" not in raw_df.columns or "subject_id" not in processed_df.columns:
        raise ValueError("Les DataFrame doivent contenir la colonne 'subject_id'.")

    feature_candidates = _collect_temporal_feature_columns(processed_df, feature_cols, time_col)
    if not feature_candidates:
        print("Aucune feature numerique temporelle disponible pour le rapport B.")
        return

    chosen_features = feature_candidates[:max_features]

    raw_stats = _temporal_subject_stats(raw_df, time_col)
    proc_stats = _temporal_subject_stats(processed_df, time_col)
    merged_stats = raw_stats.merge(proc_stats, on="subject_id", how="outer", suffixes=("_raw", "_proc")).fillna(0)
    merged_stats = merged_stats.sort_values("n_samples_proc", ascending=False).head(max_subjects)

    print("=== Diagnostic temporel (Approche B) ===")
    print(f"Sujets raw/proc: {raw_df['subject_id'].nunique()} / {processed_df['subject_id'].nunique()}")
    print(f"Features inspectees: {chosen_features}")
    if preprocess_profile.get("frequency_sampling_hz") is not None:
        print(f"Frequency sampling (Hz): {preprocess_profile.get('frequency_sampling_hz')}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (1) Nombre d'echantillons par sujet avant/apres
    ax = axes[0, 0]
    x = np.arange(len(merged_stats))
    ax.bar(x - 0.2, merged_stats["n_samples_raw"], width=0.4, label="Avant", color="#4C72B0")
    ax.bar(x + 0.2, merged_stats["n_samples_proc"], width=0.4, label="Apres", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(merged_stats["subject_id"], rotation=45, ha="right")
    ax.set_title("Densite d'echantillonnage par sujet")
    ax.set_ylabel("Nombre d'echantillons")
    ax.legend()

    # (2) Duree temporelle couverte par sujet
    ax = axes[0, 1]
    ax.bar(x - 0.2, merged_stats["duration_s_raw"], width=0.4, label="Avant", color="#4C72B0")
    ax.bar(x + 0.2, merged_stats["duration_s_proc"], width=0.4, label="Apres", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(merged_stats["subject_id"], rotation=45, ha="right")
    ax.set_title("Couverture temporelle par sujet")
    ax.set_ylabel("Duree (s)")
    ax.legend()

    # (3) Distribution des deltas de temps
    ax = axes[1, 0]
    raw_dt = raw_df.groupby("subject_id", observed=True)[time_col].apply(
        lambda s: pd.Series(np.diff(np.sort(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float))))
    )
    proc_dt = processed_df.groupby("subject_id", observed=True)[time_col].apply(
        lambda s: pd.Series(np.diff(np.sort(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float))))
    )
    raw_dt = pd.to_numeric(raw_dt, errors="coerce")
    proc_dt = pd.to_numeric(proc_dt, errors="coerce")
    raw_dt = raw_dt[(raw_dt > 0) & np.isfinite(raw_dt)]
    proc_dt = proc_dt[(proc_dt > 0) & np.isfinite(proc_dt)]
    if len(raw_dt) > 0:
        sns.kdeplot(raw_dt, ax=ax, label="Avant", color="#4C72B0", fill=True, alpha=0.2)
    if len(proc_dt) > 0:
        sns.kdeplot(proc_dt, ax=ax, label="Apres", color="#55A868", fill=True, alpha=0.2)
    ax.set_title("Distribution des pas temporels (delta t)")
    ax.set_xlabel("Delta t (s)")
    ax.set_ylabel("Densite")
    ax.legend()

    # (4) Evolution temporelle de features (sujet representatif)
    ax = axes[1, 1]
    top_subject = merged_stats.iloc[0]["subject_id"] if len(merged_stats) else str(processed_df["subject_id"].iloc[0])
    raw_sub = raw_df[raw_df["subject_id"].astype(str) == str(top_subject)].copy().sort_values(time_col)
    proc_sub = processed_df[processed_df["subject_id"].astype(str) == str(top_subject)].copy().sort_values(time_col)

    for feat in chosen_features:
        if feat in raw_sub.columns:
            ax.plot(raw_sub[time_col], raw_sub[feat], alpha=0.25, linewidth=1.0, label=f"{feat} (avant)")
        if feat in proc_sub.columns:
            ax.plot(proc_sub[time_col], proc_sub[feat], alpha=0.9, linewidth=2.0, label=f"{feat} (apres)")

    ax.set_title(f"Dynamique de signaux - sujet {top_subject}")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Valeur")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        handles = handles[:8]
        labels = labels[:8]
    ax.legend(handles, labels, loc="best", fontsize=8)

    plt.suptitle("Rapport de pretraitement temporel - Approche B", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Figure(s) complementaire(s): comparaison frequencielle (FFT) raw vs preprocessed,
    # pour chaque feature effectivement filtree.
    filtered_features = _resolve_temporal_filtered_features(feature_candidates, preprocess_profile)
    if not filtered_features:
        print("FFT non tracee: aucune feature filtree active dans preprocess_profile.")
        return

    cutoff_markers = _temporal_fft_cutoff_markers(preprocess_profile)
    fft_display_max_hz = float(preprocess_profile.get("fft_display_max_hz", 0.5) or 0.5)
    if fft_display_max_hz <= 0:
        fft_display_max_hz = 0.5
    fft_features_per_page = 4
    n_pages = int(np.ceil(len(filtered_features) / float(fft_features_per_page)))

    for page_idx in range(n_pages):
        chunk = filtered_features[page_idx * fft_features_per_page:(page_idx + 1) * fft_features_per_page]
        if not chunk:
            continue

        n_cols = 2
        n_rows = int(np.ceil(len(chunk) / float(n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.2 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for i, feat in enumerate(chunk):
            ax = axes[i]
            raw_freqs, raw_power = _compute_subject_fft_spectrum(
                raw_df,
                subject_id=top_subject,
                time_col=time_col,
                feature_cols=[feat],
            )
            proc_freqs, proc_power = _compute_subject_fft_spectrum(
                processed_df,
                subject_id=top_subject,
                time_col=time_col,
                feature_cols=[feat],
            )

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

        for i in range(len(chunk), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f"FFT par feature filtree - sujet {top_subject} (page {page_idx + 1}/{n_pages})",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        plt.show()


def plot_preprocessing_report_by_approach(
    raw_df,
    processed_df,
    feature_cols,
    preprocess_profile,
    **kwargs,
):
    """Router de visualisation de pretraitement selon l'approche (A/B).

    - Approche A -> plot_preprocessing_report
    - Approche B -> plot_temporal_preprocessing_report
    """
    approach = str((preprocess_profile or {}).get("approach", "A")).strip().upper()
    if approach == "B":
        return plot_temporal_preprocessing_report(
            raw_df=raw_df,
            processed_df=processed_df,
            feature_cols=feature_cols,
            preprocess_profile=preprocess_profile,
            **kwargs,
        )

    return plot_preprocessing_report(
        raw_df=raw_df,
        processed_df=processed_df,
        feature_cols=feature_cols,
        preprocess_profile=preprocess_profile,
    )
