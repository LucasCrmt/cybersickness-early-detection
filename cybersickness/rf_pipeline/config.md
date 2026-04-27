# Template Config Profil (copier-coller dans le notebook)

Copier le bloc Python ci-dessous dans la cellule de profils du notebook.

```python
DATA_PROFILE = {
    "source": "csv",  # csv | mat
    "file_path": r"../data/Indicateurs calculés/FullTimeIndicatorsMat_2_1.csv",
    "subject_id_col": "Participant",
    "mat_file_path": r"../data/Indicateurs calculés/FullTimeIndicatorsMinutes1.mat",
    "mat_variable": "FullTimeIndicatorsMat",
    "feature_columns": {
        0: "Time",
        1: "HMDPosX",
        2: "HMDPosY",
        3: "HMDPosZ",
        4: "RotX",
        5: "RotY",
        6: "RotZ",
        7: "SuggestedRotationX",
        8: "SuggestedRotationY",
        9: "SuggestedRotationZ",
        10: "LeftPupilDiameter",
        11: "RightPupilDiameter",
        12: "XGazeDirection",
        13: "YGazeDirection",
        14: "Confidence",
        15: "IsBoat",
        16: "XWorldPosition",
        17: "YWorldPosition",
    },
    "subject_id_count_hint": 42,
}

REPRESENTATION_PROFILE = {
    "approach": "A",  # A | B
    "drop_time_column": True,
    "aggregated_stats": ["mean", "std", "min", "max", "median", "q25", "q75", "slope"],
    "window_length": 60,
    "window_stride": 30,
}

PREPROCESS_PROFILE = {
    "clip_quantiles": [0.01, 0.99],
    "imputation_strategy": "median",
    "drop_low_information_features": True,
    "min_valid_features": 1,
}

TARGET_PROFILE = {
    "subject_id_col": "Sujet",
    "source": "xlsx",  # csv | xlsx | column_in_features
    "csv_path": r"../data/Indicateurs calculés/labels.csv",
    "xlsx_path": r"../data/Questionnaires/FMS1_org.xlsx",
    "sheet_name": "Feuil1",
    "target_col": "target",
    "target_mode": "fixed_minute",  # fixed_minute | mean_all_minutes | mean_range | last_minute
    "target_minute": 14,
    "minute_start": 1,
    "minute_end": 14,
    "minute_columns": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
}

MODEL_PROFILE = {
    "task_type": "regression",  # classification | regression
    "split_method": "random",  # group | random
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
    "output_dir": r"../outputs/random_forest_modulaire",
    "save_model_card": True,
}
```