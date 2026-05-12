import os

import numpy as np
import pandas as pd
import scipy.io

from pipeline.frequential_resampling import apply_lomb_scargle_frequency_sampling

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
    """Charge un CSV de features en utilisant uniquement les colonnes declarees dans data_profile.

    Parametres :
    data_profile: dict de configuration, avec cles obligatoires
        - 'file_path'      : chemin du CSV
        - 'subject_id_col' : nom exact de la colonne identifiant sujet dans le CSV brut
        - 'time_col'       : nom exact de la colonne temporelle dans le CSV brut

    Retour :
    DataFrame harmonise avec les colonnes standard 'subject_id' et 'time'.
    """
    csv_path = data_profile["file_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)

    if "subject_id_col" not in data_profile:
        raise ValueError("data_profile doit definir explicitement 'subject_id_col'.")
    if "time_col" not in data_profile:
        raise ValueError("data_profile doit definir explicitement 'time_col'.")

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

    time_col = data_profile["time_col"]
    if time_col not in df.columns:
        raise ValueError(
            f"Colonne temporelle introuvable: '{time_col}'. Colonnes disponibles: {list(df.columns)}"
        )
    if time_col != "time":
        df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["time"] = df["time"].astype(float)

    # minute utile pour une fusion cible per_minute.
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

    if preprocess_profile is not None and preprocess_profile.get("frequency_sampling_hz") is not None:
        df = apply_lomb_scargle_frequency_sampling(df, preprocess_profile)

    return df


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
