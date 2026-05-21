import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from pipeline.models import build_model, get_search_space


def evaluate_test_set(
    final_model,
    X_test_imp,
    y_test,
    model_profile,
    show_plots=True,
    target_profile=None,
    class_order=None,
):
    pred_test = final_model.predict(X_test_imp)

    if model_profile["task_type"] == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, pred_test),
            "precision_weighted": precision_score(y_test, pred_test, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, pred_test, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, pred_test, average="weighted", zero_division=0),
        }
        report = classification_report(y_test, pred_test, zero_division=0)

        if show_plots:
            observed = pd.concat([pd.Series(y_test), pd.Series(pred_test)], ignore_index=True)
            observed_labels = [x for x in observed.dropna().unique().tolist()]

            if class_order is not None:
                labels = [c for c in class_order if c in observed_labels]
            else:
                configured_order = None
                if target_profile is not None:
                    configured_order = (target_profile.get("discretize") or {}).get("labels")

                if configured_order is not None:
                    labels = [c for c in configured_order if c in observed_labels]
                    if len(labels) == 0:
                        # Fallback by string value when configured labels and predictions have different dtypes.
                        by_str = {}
                        for obs_label in observed_labels:
                            by_str[str(obs_label)] = obs_label
                        labels = [by_str[str(c)] for c in configured_order if str(c) in by_str]
                else:
                    labels = sorted(observed_labels, key=str)

            if len(labels) == 0:
                labels = sorted(observed_labels, key=str)

            if len(labels) > 0:
                cm = confusion_matrix(y_test, pred_test, labels=labels)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels)
                plt.title("Matrice de confusion - Test")
                plt.xlabel("Predit")
                plt.ylabel("Reel")
                plt.tight_layout()
                plt.show()

        return pred_test, metrics, report

    metrics = {
        "mae": mean_absolute_error(y_test, pred_test),
        "rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
        "r2": r2_score(y_test, pred_test),
    }

    if show_plots:
        plt.figure(figsize=(7, 7))
        plt.scatter(y_test, pred_test, alpha=0.6)
        lo = min(np.min(y_test), np.min(pred_test))
        hi = max(np.max(y_test), np.max(pred_test))
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.title("Prediction vs Verite")
        plt.xlabel("Verite")
        plt.ylabel("Prediction")
        plt.tight_layout()
        plt.show()

    return pred_test, metrics, None


def evaluate_by_subject(dataset_df, test_idx, y_test, pred_test, model_profile):
    test_df = dataset_df.iloc[test_idx].copy().reset_index(drop=True)
    test_df["y_true"] = y_test
    test_df["y_pred"] = pred_test

    if model_profile["task_type"] == "classification":
        by_subject = test_df.groupby("subject_id").apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                    "f1_weighted": f1_score(g["y_true"], g["y_pred"], average="weighted", zero_division=0),
                }
            )
        )
    else:
        by_subject = test_df.groupby("subject_id").apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                    "rmse": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                }
            )
        )

    return by_subject.reset_index()


def evaluate_robustness(final_model, X_test_imp, y_test, model_profile, eval_profile, seed=42, show_plot=True):
    rng = np.random.default_rng(seed)
    noise_scores = []

    for _ in range(eval_profile["robustness_repeats"]):
        noise = rng.normal(0, eval_profile["robustness_noise_std"], size=X_test_imp.shape)
        pred_noisy = final_model.predict(X_test_imp + noise)

        if model_profile["task_type"] == "classification":
            noise_scores.append(f1_score(y_test, pred_noisy, average="weighted", zero_division=0))
        else:
            noise_scores.append(np.sqrt(mean_squared_error(y_test, pred_noisy)))

    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(noise_scores, marker="o")
        plt.title("Stabilite sous bruit")
        plt.xlabel("Iteration")
        plt.ylabel("F1 bruit" if model_profile["task_type"] == "classification" else "RMSE bruit")
        plt.tight_layout()
        plt.show()

    return noise_scores


def train_with_optional_hyperparameter_search(
    X_train_imp,
    y_train,
    X_val_imp,
    y_val,
    model_profile,
    approach=None,
    sequence_length=None,
    n_features=None,
    verbose=True,
):
    """Entraîne un modèle avec ou sans recherche d'hyperparamètres.

    - Si `use_hyperparam_search` est True, lance une recherche sur la grille
      appropriée au `model_type`.
    - Si `use_hyperparam_search` est False, entraîne directement le modèle avec
      ses valeurs par défaut.

    Le paramètre `approach` est accepté pour contextualiser l'appel (A/B), sans
    changer la logique de sélection des hyperparamètres.

    Retourne:
        final_model, best_params, results_df
    """
    _ = approach  # Conservé pour rendre l'appel explicite côté notebook.

    use_search = bool(model_profile.get("use_hyperparam_search", True))
    is_classification = model_profile["task_type"] == "classification"
    model_type = str(model_profile.get("model_type", "random_forest")).lower()

    if model_type in {"cnn_1d", "inception_time", "bilstm", "cnn_lstm"} and (sequence_length is None or n_features is None):
        raise ValueError("sequence_length et n_features sont requis pour les modeles temporels deep.")

    if use_search:
        search_space = get_search_space(model_profile["task_type"], model_profile=model_profile)
        grid = list(ParameterGrid(search_space))
        total_trials = len(grid)

        if verbose:
            print(f"Debut recherche hyperparametres: {total_trials} combinaisons")

        rows = []
        best_score = -np.inf
        best_params = None
        t0 = time.time()

        for i, params in enumerate(grid, start=1):
            params = dict(params)
            if model_type in {"cnn_1d", "inception_time", "bilstm", "cnn_lstm"}:
                params["sequence_length"] = int(sequence_length)
                params["n_features"] = int(n_features)

            elapsed = time.time() - t0
            avg_per_trial = elapsed / max(i - 1, 1)
            eta = avg_per_trial * (total_trials - i + 1) if i > 1 else float("nan")

            if verbose:
                print(
                    f"[{i}/{total_trials}] "
                    f"elapsed={elapsed/60:.1f} min / "
                    f"eta={eta/60:.1f} min / "
                    f"params={params}"
                )

            model = build_model(params, model_profile)
            model.fit(X_train_imp, y_train)
            pred_val = model.predict(X_val_imp)

            row = dict(params)
            if is_classification:
                row["val_accuracy"] = accuracy_score(y_val, pred_val)
                row["val_f1_weighted"] = f1_score(y_val, pred_val, average="weighted", zero_division=0)
                score = row["val_f1_weighted"]
                metric_name = "val_f1_weighted"
            else:
                rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
                row["val_rmse"] = rmse
                row["val_r2"] = r2_score(y_val, pred_val)
                score = -rmse
                metric_name = "val_rmse (min)"

            rows.append(row)

            if score > best_score:
                best_score = score
                best_params = dict(params)
                if verbose:
                    print(f"    Nouveau meilleur score -> {metric_name}: {score:.4f}")
            elif verbose:
                print(f"    Score courant -> {metric_name}: {score:.4f}")

        results_df = pd.DataFrame(rows)
        sort_col = "val_f1_weighted" if is_classification else "val_rmse"
        ascending = not is_classification
        results_df = results_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
        final_params = dict(best_params or {})
        if model_type in {"cnn_1d", "inception_time", "bilstm", "cnn_lstm"}:
            final_params["sequence_length"] = int(sequence_length)
            final_params["n_features"] = int(n_features)

        final_model = build_model(final_params, model_profile)
        final_model.fit(np.vstack([X_train_imp, X_val_imp]), np.concatenate([y_train, y_val]))
        return final_model, best_params, results_df

    if verbose:
        print("Recherche hyperparametres desactivee: utilisation des valeurs par defaut.")

    params = {}
    if model_type in {"cnn_1d", "inception_time", "bilstm", "cnn_lstm"}:
        params["sequence_length"] = int(sequence_length)
        params["n_features"] = int(n_features)

    final_model = build_model(params, model_profile)
    final_model.fit(np.vstack([X_train_imp, X_val_imp]), np.concatenate([y_train, y_val]))

    results_df = pd.DataFrame([{"mode": "default_parameters_used"}])
    return final_model, params, results_df


def plot_feature_importance(final_model, feature_cols, top_n=15, model_profile=None):
    if not hasattr(final_model, "feature_importances_"):
        return None

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    _mt = (model_profile or {}).get("model_type", "random_forest").lower()
    _title = "XGBoost" if _mt == "xgboost" else "RandomForest"

    plt.figure(figsize=(10, 7))
    sns.barplot(data=imp_df.head(top_n), x="importance", y="feature", orient="h")
    plt.title(f"Top features - {_title}")
    plt.tight_layout()
    plt.show()

    return imp_df


def plot_pca_if_classification(X_test_imp, y_test, model_profile, seed=42):
    if model_profile["task_type"] != "classification":
        return

    X_for_pca = np.asarray(X_test_imp)
    if X_for_pca.ndim == 3:
        # Flatten temporal tensors (n_samples, seq_len, n_features) for PCA visualization.
        X_for_pca = X_for_pca.reshape(X_for_pca.shape[0], -1)
    elif X_for_pca.ndim != 2:
        return

    if X_for_pca.shape[1] < 2:
        return

    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X_for_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_test, palette="tab10", s=40)
    plt.title("Projection PCA - ensemble test")
    plt.tight_layout()
    plt.show()
