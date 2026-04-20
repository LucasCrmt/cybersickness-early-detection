import numpy as np
import pandas as pd
import scipy.io

DATA_DIR = 'data/'

RAW_SIGNAL_COLS = [
    'Time',
    'HMDPosX', 'HMDPosY', 'HMDPosZ',
    'RotX', 'RotY', 'RotZ',
    'Suggested Rotation X', 'Suggested Rotation Y', 'Suggested Rotation Z',
    'Left Pupil Diameter', 'Right Pupil Diameter',
    'X gaze direction', 'Y gaze direction',
    'Confidence',
    'isBoat',
    'X World Position', 'Y World Position',
]

SAMPLING_RATE = 90  # Hz

INDICATOR_COLS = [
    'Participant',
    'Amp01X', 'Amp01Y', 'Amp01Z',
    'Amp04X', 'Amp04Y', 'Amp04Z',
    'TotMovX', 'TotMovY', 'TotMovZ', 'TotMovXYZ',
    '%Pow01X', '%Pow01Y', '%Pow01Z',
    '%Pow04X', '%Pow04Y', '%Pow04Z',
    'Amp01EyeX', 'Amp01EyeY',
    'Amp04EyeX', 'Amp04EyeY',
    'TotMovEyeX', 'TotMovEyeY',
    '%Pow01EyeX', '%Pow01EyeY',
    '%Pow04EyeX', '%Pow04EyeY',
    'Ellipse95Eye',
    'Ellipse95WorldPos',
    'PupilDiamX', 'PupilDiamY',
    '%Boat',
]

INDICATOR_NUMERIC_COLS = INDICATOR_COLS[1:]

### Fonctions d'extraction et de chargement des données

def extract_workspace_strings(mat_dict):
    """Extrait les chaînes UTF-16LE depuis le workspace MATLAB embarqué."""
    ws = mat_dict.get('__function_workspace__')
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
                s = raw[i:end].decode('utf-16le', errors='replace')
                strings.append(s)
            i = end + 2
        else:
            i += 1
    return strings

def get_subject_ids(mat_dict, n_subjects=42):
    """Extrait les identifiants sujets depuis le workspace MATLAB."""
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


def cell_to_float(cell):
    """Convertit une cellule MATLAB en float (NaN si vide ou MCOS)."""
    if isinstance(cell, (int, float)):
        return float(cell)
    if hasattr(cell, 'shape'):
        if cell.size == 0:
            return np.nan
        if cell.dtype.names and 's1' in cell.dtype.names:
            return np.nan
        return float(cell.flat[0])
    return np.nan


def load_raw_data(phase=1):
    """Charge DataPhase{phase}.mat et retourne (subject_ids, raw_cell_array)."""
    path = f'{DATA_DIR}Données_brutes/DataPhase{phase}.mat'
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    data = mat[f'DataPhase{phase}']
    subject_ids = get_subject_ids(mat, n_subjects=42)
    return subject_ids, data


def get_subject_dataframe(data_cell, subject_idx):
    """Construit un DataFrame pandas pour un sujet donné à partir des données brutes."""
    ts = data_cell[subject_idx, 2]  # (75600, 18)
    df = pd.DataFrame(ts, columns=RAW_SIGNAL_COLS)
    df.index = df['Time']
    df.index.name = 'Time (s)'
    return df

### Indicateurs calculés

def load_full_indicators():
    """Charge FullTimeIndicatorsMat et retourne un dict de DataFrames {Phase1, Phase2}."""
    path = f'{DATA_DIR}Indicateurs calculés/FullTimeIndicatorsMat.mat'
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    d = mat['FullTimeIndicatorsMat']
    subject_ids = get_subject_ids(mat, n_subjects=42)

    dfs = {}
    for phase_idx, phase_name in enumerate(['Phase1', 'Phase2']):
        inner = d[1, phase_idx]  # (43, 32)
        rows = []
        for i in range(inner.shape[0]):
            row = [cell_to_float(inner[i, j]) for j in range(inner.shape[1])]
            rows.append(row)
        df = pd.DataFrame(rows, columns=INDICATOR_COLS)
        df = df.dropna(how='all').reset_index(drop=True)
        df.insert(0, 'sujet', subject_ids[:len(df)])
        df = df.drop(columns=['Participant'])
        dfs[phase_name] = df
    return dfs


def load_indicators_per_minute(phase=1):
    """Charge FullTimeIndicatorsMinutes{phase} et retourne un DataFrame long.

    Colonnes : sujet, minute, + 31 indicateurs.
    """
    path = f'{DATA_DIR}Indicateurs calculés/FullTimeIndicatorsMinutes{phase}.mat'
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    key = f'FullTimeIndicatorsMinutes{phase}'
    d = mat[key]  # (42, 14)
    subject_ids = get_subject_ids(mat, n_subjects=42)

    rows = []
    for i in range(d.shape[0]):
        for minute in range(d.shape[1]):
            cell = d[i, minute]
            if not hasattr(cell, 'shape') or cell.size == 0:
                continue
            if cell.dtype == object and len(cell) == 32:
                vals = [cell_to_float(cell[j]) for j in range(32)]
                row = {
                    'sujet': subject_ids[i] if i < len(subject_ids) else f'S{i}',
                    'minute': minute + 1,
                }
                for j, col in enumerate(INDICATOR_NUMERIC_COLS):
                    row[col] = vals[j + 1]
                rows.append(row)

    return pd.DataFrame(rows)


### RR intervals

def load_rr_intervals():
    """Charge RRintervalsClean.mat et retourne le cell array (36 sujets × 10 segments)."""
    path = f'{DATA_DIR}Données Cohérence Cardiaque/RRintervalsClean.mat'
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    return mat['RRintervalsClean']


def is_rr_cell(cell):
    """Vérifie si une cellule contient bien des intervalles RR numériques (uint16/int/float)."""
    return (hasattr(cell, 'dtype')
            and cell.dtype.kind in ('u', 'i', 'f')
            and cell.size > 0)
