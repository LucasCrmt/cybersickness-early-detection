"""Reechantillonnage frequentiel par la methode de Lomb-Scargle.

Ce fichier fournit des utilitaires pour rééchantillonner des signaux
temporels non-uniformes sur une grille régulière à une fréquence cible,
en utilisant un ajustement harmonique (cosinus + sinus) dont l'amplitude
est validée par le périodogramme de Lomb-Scargle.
"""

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

def _to_seconds(series, time_unit):
    """Convertit une série temporelle vers les secondes.

    Paramètres :
    series   : pd.Series avec valeurs temporelles
    time_unit: str, parmi 'ms' (millisecondes), 'min' (minutes), 's' (secondes)

    Retour :
    np.ndarray : valeurs converties en secondes
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if time_unit == "ms":
        return s / 1000.0
    if time_unit == "min":
        return s * 60.0
    return s


def _fit_single_frequency_signal(t, y, freq_hz):
    """Ajuste une sinusoïde à la fréquence cible sur les points d'échantillonnage.

    Utilise un ajustement harmonique y ≈ a·cos(ωt) + b·sin(ωt) + c via lstsq.
    La fréquence est validée par le périodogramme de Lomb-Scargle.

    Paramètres :
    t      : np.ndarray, instants temporels (en secondes)
    y      : np.ndarray, valeurs du signal
    freq_hz: float, fréquence cible en Hertz

    Retour :
    np.ndarray : coefficients [a, b, c] pour la reconstruction
    """
    omega = 2.0 * np.pi * float(freq_hz)

    # Lomb-Scargle sur le signal centré pour estimer la puissance au voisinage de la fréquence cible.
    y_centered = y - np.nanmean(y)
    _ = lombscargle(t, y_centered, np.asarray([omega]), precenter=False, normalize=True)

    # Ajustement harmonique y ~= a*cos(wt) + b*sin(wt) + c pour reconstruire sur grille régulière.
    X = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _reconstruct_on_grid(t_grid, freq_hz, coef):
    """Reconstruit la sinusoïde ajustée sur une grille temporelle régulière.

    Paramètres :
    t_grid : np.ndarray, instants réguliers où reconstruire (en secondes)
    freq_hz: float, fréquence en Hertz
    coef   : np.ndarray, coefficients [a, b, c] issus de _fit_single_frequency_signal

    Retour :
    np.ndarray : valeurs reconstruites sur la grille
    """
    omega = 2.0 * np.pi * float(freq_hz)
    a, b, c = coef
    return a * np.cos(omega * t_grid) + b * np.sin(omega * t_grid) + c


def _frequency_sample_group_lomb_scargle(group_df, time_col, feature_cols, freq_hz):
    """Rééchantillonne un groupe (sujet) à une fréquence donnée.

    Paramètres :
    group_df   : DataFrame d'un seul sujet, trié par colonne temporelle
    time_col   : str, nom de la colonne temporelle
    feature_cols: list de str, colonnes features à rééchantillonner
    freq_hz    : float, fréquence cible en Hertz (step = 1/freq_hz secondes)

    Retour :
    DataFrame avec grille régulière et features rééchantillonnées, ou DataFrame vide
    si le groupe est trop petit/invalide
    """
    if group_df.empty:
        return group_df

    t = pd.to_numeric(group_df[time_col], errors="coerce").to_numpy(dtype=float)
    finite_t = np.isfinite(t)
    if finite_t.sum() < 3:
        return group_df.iloc[0:0].copy()

    t = t[finite_t]
    base = group_df.loc[finite_t].copy()

    t_min = float(np.nanmin(t))
    t_max = float(np.nanmax(t))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return group_df.iloc[0:0].copy()

    step = 1.0 / float(freq_hz)
    t_grid = np.arange(t_min, t_max + step * 0.5, step, dtype=float)
    if t_grid.size == 0:
        return group_df.iloc[0:0].copy()

    out = pd.DataFrame({time_col: t_grid})

    for c in feature_cols:
        y = pd.to_numeric(base[c], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 3:
            out[c] = np.nan
            continue

        t_valid = t[valid]
        y_valid = y[valid]

        if np.unique(t_valid).size < 3:
            out[c] = np.nan
            continue

        coef = _fit_single_frequency_signal(t_valid, y_valid, freq_hz=freq_hz)
        out[c] = _reconstruct_on_grid(t_grid, freq_hz=freq_hz, coef=coef)

    return out


def apply_lomb_scargle_frequency_sampling(df, preprocess_profile):
    """Rééchantillonne chaque sujet sur une grille temporelle régulière.

    Produit un DataFrame où chaque sujet est rééchantillonné à une (ou plusieurs)
    fréquence(s) cible(s), via ajustement harmonique + grille régulière.

        Paramètres :
        df                : DataFrame AVANT add_target
        preprocess_profile: dict de config avec cles obligatoires
                - 'frequency_sampling_hz' : float ou list[float] (OBLIGATOIRE, sinon retourne df inchangé)
                - 'time_col'              : nom exact de la colonne temporelle
                - 'subject_id_col'        : nom exact de la colonne identifiant sujet
            cles optionnelles :
                - 'time_unit'             : 'ms'|'min'|'s' (défaut: 's')
                - 'frequency_feature_cols': liste explicite de colonnes features a resampler

    Retour :
    DataFrame rééchantillonné avec colonnes "sampling_hz" et "minute" ajoutées/recalculées.
    Si preprocess_profile['frequency_sampling_hz'] est None, retourne df inchangé.
    """
    frequencies = preprocess_profile.get("frequency_sampling_hz")
    if frequencies is None:
        return df

    if isinstance(frequencies, (int, float)):
        frequencies = [float(frequencies)]
    else:
        frequencies = [float(f) for f in frequencies]

    if len(frequencies) == 0:
        raise ValueError("preprocess_profile['frequency_sampling_hz'] est vide.")
    if any(f <= 0 for f in frequencies):
        raise ValueError("Toutes les frequences de 'frequency_sampling_hz' doivent etre > 0.")

    if "time_col" not in preprocess_profile:
        raise ValueError("preprocess_profile doit definir explicitement 'time_col'.")
    if "subject_id_col" not in preprocess_profile:
        raise ValueError("preprocess_profile doit definir explicitement 'subject_id_col'.")

    out = df.copy()
    time_col = preprocess_profile["time_col"]
    if time_col not in out.columns:
        raise ValueError(
            f"Colonne temporelle introuvable: '{time_col}'. Colonnes disponibles: {list(out.columns)}"
        )

    time_unit = str(preprocess_profile.get("time_unit", "s")).strip().lower()
    out[time_col] = _to_seconds(out[time_col], time_unit=time_unit)
    out = out.dropna(subset=[time_col]).copy()

    id_col = preprocess_profile["subject_id_col"]
    if id_col not in out.columns:
        raise ValueError(
            f"Colonne subject_id introuvable: '{id_col}'. Colonnes disponibles: {list(out.columns)}"
        )

    reserved = {
        "target",
        "subject_id",
        "Participant",
        "participant",
        "row_id",
        "window_start",
        "window_end",
        "minute",
        time_col,
        "sampling_hz",
    }

    explicit_features = preprocess_profile.get("frequency_feature_cols")
    if explicit_features is not None:
        missing = [c for c in explicit_features if c not in out.columns]
        if missing:
            raise ValueError(f"Colonnes absentes dans frequency_feature_cols: {missing}")
        feature_cols = list(explicit_features)
    else:
        feature_cols = [
            c
            for c in out.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(out[c])
        ]

    if len(feature_cols) == 0:
        raise ValueError("Aucune feature numerique disponible pour le decoupage frequentiel Lomb-Scargle.")

    sampled_frames = []
    if id_col is None:
        groups_iter = [(None, out.sort_values(time_col))]
    else:
        groups_iter = [(sid, g.sort_values(time_col)) for sid, g in out.groupby(id_col, sort=False)]

    for sid, g in groups_iter:
        for f_hz in frequencies:
            g_sampled = _frequency_sample_group_lomb_scargle(g, time_col, feature_cols, freq_hz=f_hz)
            if g_sampled.empty:
                continue
            if id_col is not None:
                g_sampled[id_col] = sid
            g_sampled["sampling_hz"] = float(f_hz)
            sampled_frames.append(g_sampled)

    if len(sampled_frames) == 0:
        raise ValueError("Le decoupage Lomb-Scargle n'a produit aucune ligne exploitable.")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled["minute"] = np.floor(sampled[time_col] / 60.0).astype(int)
    if id_col != "subject_id":
        sampled = sampled.rename(columns={id_col: "subject_id"})

    if "row_id" not in sampled.columns:
        sampled["row_id"] = np.arange(len(sampled), dtype=int)

    return sampled
