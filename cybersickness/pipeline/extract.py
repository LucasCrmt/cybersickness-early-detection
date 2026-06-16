import os

import numpy as np
import pandas as pd
import scipy.io

from pipeline.frequential_resampling import (
    apply_lomb_scargle_frequency_sampling,
    apply_uniform_time_step_sampling,
    _to_seconds,
)

# MAT helpers

def extract_workspace_strings(mat_dict):
    ws = mat_dict.get("__function_workspace__")
    if ws is None:
        return []
    raw = ws.tobytes()
    strings = []
    i = 0
    while i < len(raw) - 1:
        if raw[i + 1] == 0 and 32 <= raw[i] <= 126:
            end = i
            while end < len(raw) - 1 and raw[end + 1] == 0 and 32 <= raw[end] <= 126:
                end += 2
            if end - i >= 4:
                s = raw[i:end].decode("utf-16le", errors="replace")
                strings.append(s)
            i = end + 2
        else:
            i += 1
    return strings


def get_subject_ids(mat_dict, n_subjects=42):
    strings = extract_workspace_strings(mat_dict)
    subject_ids = []
    seen = set()
    for s in strings:
        if len(s) >= 2 and all(ord(c) < 128 for c in s) and s not in seen:
            seen.add(s)
            subject_ids.append(s)
            if len(subject_ids) == n_subjects:
                break
    return subject_ids


def load_mat_matrix(data_profile):
    mat_path = data_profile.get("mat_file_path") or data_profile.get("file_path")
    mat_var = data_profile["mat_variable"]

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Fichier introuvable: {mat_path}")

    mat_dict = scipy.io.loadmat(mat_path)
    if mat_var not in mat_dict:
        raise KeyError(f"Variable '{mat_var}' absente du fichier .mat")

    data_matrix = np.asarray(mat_dict[mat_var], dtype=object)
    subject_ids = get_subject_ids(mat_dict, n_subjects=data_profile.get("subject_id_count_hint", 42))
    return mat_dict, data_matrix, subject_ids


# CSV features

def load_csv_features(data_profile):
    """Charge un CSV de features.

    Parametres :
    data_profile: dict de configuration avec :
        - 'file_path'      : chemin du CSV (obligatoire)
        - 'subject_id_col' : nom de la colonne sujet dans le CSV (obligatoire)
        - 'time_col'       : nom de la colonne de temps en secondes (approche B uniquement,
                             optionnel). Si presente, elle est renommee 'time' et une colonne
                             'minute' est derivee via floor(time / 60).

    Retour :
    DataFrame avec au moins les colonnes 'subject_id' et 'row_id'.
    Approche B : colonnes supplementaires 'time' et 'minute'.
    """
    csv_path = data_profile["file_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)

    if "subject_id_col" not in data_profile:
        raise ValueError("data_profile doit definir explicitement 'subject_id_col'.")

    sid_col = data_profile["subject_id_col"]
    if sid_col not in df.columns:
        raise ValueError(
            f"Colonne subject_id introuvable: '{sid_col}'. Colonnes disponibles: {list(df.columns)}"
        )
    if sid_col != "subject_id":
        df = df.rename(columns={sid_col: "subject_id"})

    df["subject_id"] = df["subject_id"].astype(str).str.strip()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df), dtype=int)

    # Approche B : colonne de temps en secondes → 'time' + 'minute' derivee
    time_col = data_profile.get("time_col")
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(
                f"Colonne temporelle introuvable: '{time_col}'. Colonnes disponibles: {list(df.columns)}"
            )
        if time_col != "time":
            df = df.rename(columns={time_col: "time"})
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).copy()
        df["time"] = df["time"].astype(float)

        if "minute" not in df.columns:
            df["minute"] = np.floor(df["time"] / 60.0).astype(int)

    return df

def load_and_resample_features(data_profile, preprocess_profile=None):
    """Charge les features brutes et applique le reechantillonnage Lomb-Scargle si configure.

    A utiliser pour l'approche B (series temporelles) avant l'appel a add_target,
    de sorte que le frequency sampling soit realise sur le signal brut sans
    perturber la colonne cible.

    Pour l'approche A (indicateurs agrégés), utiliser directement load_csv_features.
    """
    df = load_csv_features(data_profile)

    if preprocess_profile is not None and preprocess_profile.get("use_frequency_resampling", False):
        method = str(preprocess_profile.get("frequency_resampling_method", "lomb_scargle")).strip().lower()
        if method == "lomb_scargle":
            if preprocess_profile.get("frequency_sampling_hz") is None:
                raise ValueError(
                    "'frequency_sampling_hz' est requis quand frequency_resampling_method='lomb_scargle'."
                )
            df = apply_lomb_scargle_frequency_sampling(df, preprocess_profile)
        elif method == "uniform_time_step":
            df = apply_uniform_time_step_sampling(df, preprocess_profile)
        else:
            raise ValueError(
                "frequency_resampling_method invalide. Valeurs acceptees: 'lomb_scargle', 'uniform_time_step'."
            )

    return df


def _resolve_window_duration_seconds(preprocess_profile):
    duration = preprocess_profile.get("window_duration_s")
    if duration is None:
        return None

    duration = float(duration)
    if duration <= 0:
        raise ValueError("'window_duration_s' doit etre > 0.")
    return duration


def _resolve_window_overlap_seconds(preprocess_profile, window_duration_s):
    if preprocess_profile.get("window_overlap_s") is not None:
        overlap = float(preprocess_profile.get("window_overlap_s"))
    else:
        ratio = preprocess_profile.get("window_overlap_ratio")
        if ratio is None:
            overlap = 0.0
        else:
            ratio = float(ratio)
            if not 0 <= ratio < 1:
                raise ValueError("'window_overlap_ratio' doit etre dans [0, 1[.")
            overlap = float(window_duration_s) * ratio

    if overlap < 0:
        raise ValueError("'window_overlap_s' doit etre >= 0.")
    if overlap >= window_duration_s:
        raise ValueError("'window_overlap_s' doit etre strictement inferieur a 'window_duration_s'.")
    return overlap


def _estimate_sampling_step_seconds(df, preprocess_profile):
    configured = preprocess_profile.get("uniform_time_step_s")
    if configured is not None:
        step = float(configured)
        if step <= 0:
            raise ValueError("'uniform_time_step_s' doit etre > 0.")
        return step

    if "sampling_hz" in df.columns:
        sampling_hz = pd.to_numeric(df["sampling_hz"], errors="coerce").dropna()
        if len(sampling_hz) > 0:
            step = float(1.0 / np.nanmedian(sampling_hz.to_numpy(dtype=float)))
            if np.isfinite(step) and step > 0:
                return step

    time_col = preprocess_profile.get("time_col")
    subject_col = preprocess_profile.get("subject_id_col")
    if time_col is None or subject_col is None or time_col not in df.columns or subject_col not in df.columns:
        raise ValueError(
            "Impossible d'estimer un pas temporel: renseigner 'uniform_time_step_s' ou garder 'time_col'/'subject_id_col'."
        )

    deltas = []
    for _, g in df.groupby(subject_col, sort=False):
        t = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
        if t.size < 2:
            continue
        dt = np.diff(t)
        dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.size > 0:
            deltas.append(dt)

    if len(deltas) == 0:
        raise ValueError("Impossible d'estimer un pas temporel pour construire les fenetres.")

    all_dt = np.concatenate(deltas)
    step = float(np.median(all_dt))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Pas temporel estime invalide.")
    return step


def apply_sliding_window_sequences(df, preprocess_profile):
    """Decoupe les series temporelles en fenetres chevauchantes et a longueur fixe.

    Chaque fenetre est reconstruite sur une grille reguliere puis aplatie en colonnes
    de la forme `feature__t0000`, `feature__t0001`, etc. Le retour reste donc un
    DataFrame tabulaire, mais chaque ligne represente une fenetre temporelle.
    """
    if "time_col" not in preprocess_profile:
        raise ValueError("preprocess_profile doit definir explicitement 'time_col'.")
    if "subject_id_col" not in preprocess_profile:
        raise ValueError("preprocess_profile doit definir explicitement 'subject_id_col'.")

    window_duration_s = _resolve_window_duration_seconds(preprocess_profile)
    if window_duration_s is None:
        return df

    out = df.copy()
    time_col = preprocess_profile["time_col"]
    subject_col = preprocess_profile["subject_id_col"]

    if time_col not in out.columns:
        raise ValueError(f"Colonne temporelle introuvable: '{time_col}'. Colonnes disponibles: {list(out.columns)}")
    if subject_col not in out.columns:
        raise ValueError(f"Colonne subject_id introuvable: '{subject_col}'. Colonnes disponibles: {list(out.columns)}")

    time_unit = str(preprocess_profile.get("time_unit", "s")).strip().lower()
    out[time_col] = _to_seconds(out[time_col], time_unit=time_unit)
    out = out.dropna(subset=[time_col]).copy()

    reserved = {
        "target",
        "subject_id",
        "Participant",
        "participant",
        "row_id",
        "window_start",
        "window_end",
        "window_center",
        "window_id",
        "minute",
        time_col,
        "sampling_hz",
    }
    explicit_features = preprocess_profile.get("window_feature_cols") or preprocess_profile.get("frequency_feature_cols")
    if explicit_features is not None:
        missing = [c for c in explicit_features if c not in out.columns]
        if missing:
            raise ValueError(f"Colonnes absentes dans window_feature_cols: {missing}")
        feature_cols = list(explicit_features)
    else:
        feature_cols = [c for c in out.columns if c not in reserved and pd.api.types.is_numeric_dtype(out[c])]

    if len(feature_cols) == 0:
        raise ValueError("Aucune feature numerique disponible pour le fenetrage temporel.")

    step_s = _estimate_sampling_step_seconds(out, preprocess_profile)
    window_points = int(round(window_duration_s / step_s))
    if window_points < 2:
        raise ValueError("La fenetre calculee contient moins de 2 points. Augmente 'window_duration_s' ou reduis le pas temporel.")

    overlap_s = _resolve_window_overlap_seconds(preprocess_profile, window_duration_s)
    overlap_points = int(round(overlap_s / step_s))
    stride_points = max(1, window_points - overlap_points)

    windowed_frames = []
    for sid, g in out.groupby(subject_col, sort=False):
        g = g.sort_values(time_col).copy()
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(t)
        if finite.sum() < window_points:
            continue

        g = g.loc[finite].copy()
        t = t[finite]
        order = np.argsort(t)
        t = t[order]
        g = g.iloc[order].copy()

        uniq_t, uniq_idx = np.unique(t, return_index=True)
        if uniq_t.size < window_points:
            continue
        g = g.iloc[uniq_idx].copy()

        t_min = float(uniq_t[0])
        t_max = float(uniq_t[-1])
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            continue

        window_id = 0
        start = t_min
        while start + window_duration_s <= t_max + step_s * 0.5:
            end = start + window_duration_s
            mask = (t >= start) & (t <= end + step_s * 0.5)
            if mask.sum() >= window_points:
                local = g.loc[mask, [time_col] + feature_cols].copy()
                local[time_col] = pd.to_numeric(local[time_col], errors="coerce")
                local = local.dropna(subset=[time_col]).sort_values(time_col)
                local = local.groupby(time_col, as_index=False).mean(numeric_only=True)

                t_local = local[time_col].to_numpy(dtype=float)
                if t_local.size >= window_points:
                    t_grid = start + np.arange(window_points, dtype=float) * step_s
                    row = {
                        subject_col: sid,
                        "window_id": window_id,
                        "window_start": float(start),
                        "window_end": float(end),
                        "window_center": float(start + window_duration_s / 2.0),
                        "window_duration_s": float(window_duration_s),
                        "window_overlap_s": float(overlap_s),
                        "sampling_hz": float(1.0 / step_s),
                        "minute": int(np.floor((start + window_duration_s / 2.0) / 60.0)),
                    }

                    if "target" in g.columns:
                        center_t = start + window_duration_s / 2.0
                        local_target = g.loc[mask, [time_col, "target"]].copy()
                        local_target[time_col] = pd.to_numeric(local_target[time_col], errors="coerce")
                        local_target = local_target.dropna(subset=[time_col, "target"]).sort_values(time_col)
                        if not local_target.empty:
                            nearest_idx = (local_target[time_col] - center_t).abs().idxmin()
                            row["target"] = local_target.loc[nearest_idx, "target"]
                        else:
                            # Fallback robuste: premiere valeur cible disponible sur le sujet.
                            subj_target = pd.Series(g.get("target")).dropna()
                            if len(subj_target) > 0:
                                row["target"] = subj_target.iloc[0]

                    for feat in feature_cols:
                        y = pd.to_numeric(local[feat], errors="coerce").to_numpy(dtype=float)
                        good = np.isfinite(y)
                        if good.sum() < 2:
                            values = np.full(window_points, np.nan, dtype=float)
                        else:
                            values = np.interp(t_grid, t_local[good], y[good])

                        for step_idx, value in enumerate(values):
                            row[f"{feat}__t{step_idx:04d}"] = float(value)

                    windowed_frames.append(row)
                    window_id += 1

            start += stride_points * step_s

    if len(windowed_frames) == 0:
        raise ValueError("Le fenetrage temporel n'a produit aucune fenetre exploitable.")

    return pd.DataFrame(windowed_frames)


def load_features_for_approach(data_profile, preprocess_profile=None, verbose=True):
    """Charge les features selon l'approche (A/B) et le mode de reechantillonnage.

    Règles :
    - Approche A : chargement CSV standard
    - Approche B + use_frequency_resampling=True + frequency_sampling_hz défini :
      chargement puis reechantillonnage Lomb-Scargle
    - Approche B sans reechantillonnage : chargement CSV standard en conservant la dimension temporelle
        - Si 'window_duration_s' est defini, le fenetrage peut etre effectue a l'etape
            extract (window_stage='extract') ou differe au post-pretraitement
            (window_stage='post_preprocess', valeur par defaut).
    """
    preprocess_profile = preprocess_profile or {}
    approach = str(preprocess_profile.get("approach", "A")).strip().upper()
    if approach not in {"A", "B"}:
        raise ValueError("preprocess_profile['approach'] doit etre 'A' ou 'B'.")

    use_resampling = bool(preprocess_profile.get("use_frequency_resampling", False))
    method = str(preprocess_profile.get("frequency_resampling_method", "lomb_scargle")).strip().lower()
    has_sampling = preprocess_profile.get("frequency_sampling_hz") is not None

    if approach == "B":
        if use_resampling and (method == "uniform_time_step" or has_sampling):
            if verbose:
                if method == "uniform_time_step":
                    print("Approche B activee avec uniformisation du pas temporel sur les donnees brutes.")
                else:
                    print("Approche B activee avec frequency sampling (Lomb-Scargle) sur les donnees brutes.")
            df = load_and_resample_features(data_profile, preprocess_profile)
        else:
            if verbose:
                print("Approche B activee sans reechantillonnage: series temporelles brutes conservees.")
            df = load_csv_features(data_profile)

        window_stage = str(preprocess_profile.get("window_stage", "post_preprocess")).strip().lower()
        if preprocess_profile.get("window_duration_s") is not None and window_stage == "extract":
            if verbose:
                overlap_desc = preprocess_profile.get("window_overlap_s")
                if overlap_desc is None and preprocess_profile.get("window_overlap_ratio") is not None:
                    overlap_desc = f"ratio={preprocess_profile.get('window_overlap_ratio')}"
                print(
                    "Fenetrage temporel active: ",
                    f"window_duration_s={preprocess_profile.get('window_duration_s')}",
                    f", overlap={overlap_desc if overlap_desc is not None else 0}",
                )
            return apply_sliding_window_sequences(df, preprocess_profile)

        if preprocess_profile.get("window_duration_s") is not None and window_stage != "extract" and verbose:
            print("Fenetrage differe: il sera applique apres le pretraitement (window_stage='post_preprocess').")

        return df

    if verbose:
        print("Approche A activee: chargement tabulaire standard.")
    return load_csv_features(data_profile)


# Target helpers

def _normalize_col_name(x):
    return str(x).strip().lower()


def _resolve_column(df, expected_name):
    if expected_name in df.columns:
        return expected_name

    norm_expected = _normalize_col_name(expected_name)
    for c in df.columns:
        if _normalize_col_name(c) == norm_expected:
            return c

    raise KeyError(f"Colonne '{expected_name}' introuvable. Colonnes disponibles: {list(df.columns)}")


def _minute_number_from_col(col):
    s = str(col).strip()
    if s.isdigit():
        return int(s)

    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return int(digits)

    return None


def _resolve_minute_columns(df, target_profile):
    requested = target_profile.get("minute_columns")

    if requested is None:
        resolved = []
        for c in df.columns:
            m = _minute_number_from_col(c)
            if m is not None:
                resolved.append((m, c))
        resolved = sorted(set(resolved), key=lambda x: x[0])
        return [c for _, c in resolved], {c: m for m, c in resolved}

    resolved_cols = []
    minute_map = {}

    for req in requested:
        if req in df.columns:
            col = req
        else:
            s = str(req)
            if s in df.columns:
                col = s
            else:
                try:
                    i = int(req)
                except Exception:
                    i = None
                if i is not None and i in df.columns:
                    col = i
                else:
                    col = _resolve_column(df, s)

        resolved_cols.append(col)
        m = _minute_number_from_col(col)
        if m is not None:
            minute_map[col] = m

    return resolved_cols, minute_map


def _build_target_from_minutes(target_df, target_profile, sid_col):
    minute_cols, minute_map = _resolve_minute_columns(target_df, target_profile)
    if len(minute_cols) == 0:
        raise ValueError("Aucune colonne minute detectee pour construire la cible.")

    minute_values = target_df[minute_cols].apply(pd.to_numeric, errors="coerce")
    mode = target_profile.get("target_mode", "fixed_minute")

    if mode == "fixed_minute":
        wanted = int(target_profile.get("target_minute", 14))
        selected = [c for c in minute_cols if minute_map.get(c) == wanted]
        if len(selected) == 0:
            raise ValueError(
                f"Minute cible {wanted} introuvable. Minutes disponibles: {sorted(set(minute_map.values()))}"
            )

        preferred = minute_values[selected[0]]
        ordered_cols = sorted(minute_cols, key=lambda c: minute_map.get(c, float("inf")))
        last_available = minute_values[ordered_cols].ffill(axis=1).iloc[:, -1]
        target_series = preferred.where(preferred.notna(), last_available)

    elif mode == "mean_all_minutes":
        target_series = minute_values.mean(axis=1, skipna=True)

    elif mode == "mean_range":
        m_start = int(target_profile.get("minute_start", 1))
        m_end = int(target_profile.get("minute_end", 14))
        selected = [c for c in minute_cols if minute_map.get(c) is not None and m_start <= minute_map[c] <= m_end]
        if len(selected) == 0:
            raise ValueError(
                f"Aucune minute dans l'intervalle [{m_start}, {m_end}]. Minutes disponibles: {sorted(set(minute_map.values()))}"
            )
        target_series = minute_values[selected].mean(axis=1, skipna=True)

    elif mode == "last_minute":
        available = [(minute_map.get(c), c) for c in minute_cols if minute_map.get(c) is not None]
        if len(available) == 0:
            raise ValueError("Impossible de determiner la derniere minute disponible.")
        last_col = sorted(available, key=lambda x: x[0])[-1][1]
        target_series = minute_values[last_col]

    elif mode == "per_minute":
        # une ligne par (sujet, minute)
        tmp = minute_values.copy()
        tmp.insert(0, "subject_id", target_df[sid_col].astype(str).values)
        long = tmp.melt(id_vars="subject_id", value_vars=minute_cols, var_name="_col", value_name="target")
        long["minute"] = long["_col"].map(lambda c: minute_map.get(c))
        long = long.dropna(subset=["subject_id", "target", "minute"])
        long["minute"] = long["minute"].astype(int)
        return long[["subject_id", "minute", "target"]].reset_index(drop=True)

    else:
        raise ValueError(
            "target_mode doit etre 'fixed_minute', 'mean_all_minutes', 'mean_range', 'last_minute' ou 'per_minute'."
        )

    out = pd.DataFrame({"subject_id": target_df[sid_col].astype(str), "target": target_series})
    out = out.dropna(subset=["subject_id", "target"])
    return out


def _build_target_table(target_df, target_profile):
    sid = _resolve_column(target_df, target_profile["subject_id_col"])

    tgt_name = target_profile.get("target_col")
    if tgt_name is not None:
        try:
            tgt = _resolve_column(target_df, tgt_name)
            t = target_df[[sid, tgt]].copy()
            t = t.rename(columns={sid: "subject_id", tgt: "target"})
            t["subject_id"] = t["subject_id"].astype(str)
            t["target"] = pd.to_numeric(t["target"], errors="coerce")
            t = t.dropna(subset=["subject_id", "target"])
            if len(t) > 0:
                return t
        except KeyError:
            pass

    return _build_target_from_minutes(target_df, target_profile, sid)


def load_per_minute_targets(target_profile):
    """Charge les cibles par minute sans merger avec un DataFrame de features.

    Utile pour les pipelines sur données brutes où les features sont construites
    séparément (ex: segmentation par minute).

    Supporte target_mode='per_minute' et target_mode='fixed_minute'.
    Pour 'fixed_minute', la cible unique par sujet est broadcastée sur toutes
    les minutes déclarées dans minute_columns, afin que segment_sequences_sliding_window
    puisse associer chaque fenêtre à la bonne cible.

    Returns:
        DataFrame avec colonnes [subject_id, minute, target] après discretisation
        si configurée dans target_profile.
    """
    source = target_profile["source"]
    target_mode = target_profile.get("target_mode", "per_minute")

    if source == "xlsx":
        xlsx_path = target_profile["xlsx_path"]
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"Fichier cible introuvable: {xlsx_path}")
        raw_df = pd.read_excel(xlsx_path, sheet_name=target_profile.get("sheet_name", 0))
    elif source == "csv":
        csv_path = target_profile["csv_path"]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier cible introuvable: {csv_path}")
        raw_df = pd.read_csv(csv_path)
    else:
        raise ValueError("load_per_minute_targets supporte uniquement 'xlsx' et 'csv'.")

    if target_mode == "per_minute":
        t = _build_target_table(raw_df, target_profile)
        t["subject_id"] = t["subject_id"].astype(str).str.strip()
        return t

    if target_mode == "fixed_minute":
        # Obtenir une cible unique par sujet via _build_target_table (retourne [subject_id, target])
        t = _build_target_table(raw_df, target_profile)
        t["subject_id"] = t["subject_id"].astype(str).str.strip()
        # Broadcaster sur toutes les minutes déclarées
        minutes = [int(m) for m in target_profile.get("minute_columns", list(range(1, 15)))]
        rows = []
        for _, row in t.iterrows():
            for m in minutes:
                rows.append({"subject_id": row["subject_id"], "minute": m, "target": row["target"]})
        return pd.DataFrame(rows)[["subject_id", "minute", "target"]]

    raise ValueError(
        f"load_per_minute_targets supporte target_mode 'per_minute' et 'fixed_minute', "
        f"reçu: '{target_mode}'."
    )


def add_target(features_df, target_profile):
    source = target_profile["source"]

    if source == "csv":
        csv_path = target_profile["csv_path"]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier cible introuvable: {csv_path}")
        target_df = pd.read_csv(csv_path)

    elif source == "xlsx":
        xlsx_path = target_profile["xlsx_path"]
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"Fichier cible introuvable: {xlsx_path}")
        target_df = pd.read_excel(xlsx_path, sheet_name=target_profile.get("sheet_name", 0))

    elif source == "column_in_features":
        tgt = target_profile["target_col"]
        if tgt not in features_df.columns:
            raise ValueError(f"Colonne cible introuvable dans features_df: {tgt}")
        merged = features_df.copy()
        if tgt != "target":
            merged = merged.rename(columns={tgt: "target"})
        return merged.dropna(subset=["target"])

    else:
        raise ValueError("TARGET_PROFILE['source'] doit etre 'csv', 'xlsx' ou 'column_in_features'.")

    t = _build_target_table(target_df, target_profile)
    merged = features_df.copy()
    merged["subject_id"] = merged["subject_id"].astype(str).str.strip()
    t["subject_id"] = t["subject_id"].astype(str).str.strip()

    if "minute" in t.columns:
        minute_col = target_profile.get("minute_col", "minute")
        if minute_col not in merged.columns:
            raise ValueError(
                f"Colonne minute '{minute_col}' introuvable dans les features. "
                "Ajoute 'minute_col' dans TARGET_PROFILE pour pointer vers la colonne minute des features."
            )
        merged["minute"] = merged[minute_col].astype(int)
        merged = merged.merge(t, on=["subject_id", "minute"], how="inner")
    else:
        merged = merged.merge(t, on="subject_id", how="inner")

    if merged.empty:
        raise ValueError(
            "Aucune ligne apres fusion features/cible. Verifie le format de subject_id dans les fichiers source."
        )

    return merged
