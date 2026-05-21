# Template Config Profil (copier-coller dans le notebook)

Copier le bloc Python ci-dessous dans la cellule de profils du notebook.

```python
DATA_PROFILE = {
    "source": "csv",  # csv / mat
    "file_path": r"../data/Indicateurs calculés/FullTimeIndicatorsMat_2_1.csv",
    "subject_id_col": "Participant",
}

PREPROCESS_PROFILE = {
    "clip_quantiles": [0.01, 0.99],
    "imputation_strategy": "median",
    "drop_low_information_features": True,
    "min_valid_features": 1,
    "normalization": None,  # None / "standard" / "minmax"

    # Approche temporelle (B) : filtres optionnels par sujet
    # "approach": "B",
    # "time_col": "time",
    # "subject_id_col": "subject_id",
    #
    # Filtre passe-bas
    # "apply_temporal_lowpass": True,
    # "lowpass_cutoff_hz": 0.05,
    # "lowpass_order": 4,
    # "lowpass_min_points": 16,
    # "lowpass_features": "all",  # "all" ou liste de noms
    #
    # Filtre passe-haut
    # "apply_temporal_highpass": False,
    # "highpass_cutoff_hz": 0.01,
    # "highpass_order": 4,
    # "highpass_min_points": 16,
    # "highpass_features": "all",  # "all" ou liste de noms
    #
    # Filtre passe-bande
    # "apply_temporal_bandpass": False,
    # "bandpass_low_cutoff_hz": 0.01,
    # "bandpass_high_cutoff_hz": 0.20,
    # "bandpass_order": 4,
    # "bandpass_min_points": 16,
    # "bandpass_features": "all",  # "all" ou liste de noms
    
    # Agrégation de colonnes (optionnel)
    # Agrégation pairwise: chaque règle fusionne exactement 2 colonnes,
    # ligne par ligne (pas de moyenne sur une liste de 3+ colonnes).
    # Exemple : fusionner Left/Right Pupil Diameter en pupil_diameter_avg
    # "column_aggregations": {
    #     "pupil_diameter_avg": {
    #         "columns": ["Left Pupil Diameter", "Right Pupil Diameter"],
    #         "strategy": "mean"  # mean | min | max | std | sum
    #     },
    #     "gaze_xy_std": {
    #         "columns": ["X gaze direction", "Y gaze direction"],
    #         "strategy": "std"
    #     }
    # },
}

TARGET_PROFILE = {
    "subject_id_col": "Sujet",
    "source": "xlsx",  # xlsx
    "xlsx_path": r"../data/Questionnaires/FMS1_org.xlsx",
    "sheet_name": "Feuil1",
    "target_col": "target",
    "target_mode": "fixed_minute",  # fixed_minute / mean_all_minutes / mean_range / last_minute / per_minute
    "target_minute": 14,            # utilisé par fixed_minute
    "minute_start": 1,              # utilisé par mean_range
    "minute_end": 14,               # utilisé par mean_range
    "minute_columns": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    # per_minute uniquement : nom de la colonne minute dans les features (défaut "minute")
    # "minute_col": "minute",

    # Discrétisation (optionnelle, pour classification)
    # Requiert task_type = "classification" dans MODEL_PROFILE
    # bins  : bornes des intervalles (N+1 valeurs pour N classes)
    # include_lowest=True -> la borne inférieure est incluse dans la 1re classe
    # labels: noms des classes, dans l'ordre croissant des bins
    #
    # Exemple 3 classes (low 0-5 / medium 6-10 / high 11-20) :
    # "discretize": {
    #     "bins":   [0, 5, 10, 20],
    #     "labels": ["low", "medium", "high"],
    # },
    #
    # Exemple 4 classes (low 0-4 / medium 5-8 / high 9-12 / very_high 13-20) :
    # "discretize": {
    #     "bins":   [0, 4, 8, 12, 20],
    #     "labels": ["low", "medium", "high", "very_high"],
    # },
}

MODEL_PROFILE = {
    "task_type": "regression",  # classification / regression
    "model_type": "random_forest",  # random_forest / xgboost / SVM
    "split_method": "random",  # group / random
    "test_size": 0.20,
    "val_size": 0.20,
    "class_weight": "balanced",
    "random_state": SEED,
}

EVAL_PROFILE = {
    "robustness_noise_std": 0.01,
    "robustness_repeats": 5,
}

OUTPUT_PROFILE = {
    "output_dir": r"../data/outputs/random_forest_modulaire",
    "save_model_card": True,
    "save_visual_report": True,
    "visual_report_format": "pdf",  # pdf / png / both
    # "all" pour tout, ou une liste explicite de fonctions
    "visual_report_functions": [
        "visual_cover_page",
        "visual_model_architecture_page",
        "visual_split_report",
        "visual_correlation_pages",
        "visual_violin_pages",
        "visual_clipping_boxplots",
        "visual_confusion_matrix", # classification uniquement
        "visual_metrics_bar",
        "visual_feature_importance",
        "visual_pca", # classification uniquement
    ],
    # Nom du fichier
    "visual_report_name": "visual_report",
    # Nb features par graph
    "max_corr_features": 32,
    "max_violin_features": 48,
    "violin_features_per_page": 9,
    # FFT temporelle (approche B)
    "max_fft_features": 12,
    "fft_features_per_page": 4,
    "top_n_importance": 20, # nombre de features dans le graphe d'importance
    # Texte libre presente sur la page de garde du rapport visuel (optionnel)
    "hypothesis": (
        """
            Hypothèse : ...
        """
    ),
}
```

## Données indicateurs calculés
```python
[
    "Amp01X",
    "Amp01Y",
    "Amp01Z",
    "Amp04X",
    "Amp04Y", 
    "Amp04Z",
    "TotMovX",
    "TotMovY",
    "TotMovZ",
    "TotMovXYZ",
    "%Pow01X",
    "%Pow01Y",
    "%Pow01Z",
    "%Pow04X",
    "%Pow04Y",
    "%Pow04Z",
    "Amp01EyeX",
    "Amp01EyeY",
    "Amp04EyeX",
    "Amp04EyeY",
    "TotMovEyeX",
    "TotMovEyeY",
    "%Pow01EyeX",
    "%Pow01EyeY",
    "%Pow04EyeX",
    "%Pow04EyeY",
    "Ellipse95Eye",
    "Ellipse95WorldPos",
    "PupilDiamX",
    "PupilDiamY",
    "%Boat",
]
```