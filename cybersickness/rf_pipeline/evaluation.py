import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
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


def evaluate_test_set(final_model, X_test_imp, y_test, model_profile, show_plots=True):
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
            labels = sorted(set(y_test) | set(pred_test), key=str)
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


def plot_feature_importance(final_model, feature_cols, top_n=15):
    if not hasattr(final_model, "feature_importances_"):
        return None

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=imp_df.head(top_n), x="importance", y="feature", orient="h")
    plt.title("Top features - RandomForest")
    plt.tight_layout()
    plt.show()

    return imp_df


def plot_pca_if_classification(X_test_imp, y_test, model_profile, seed=42):
    if model_profile["task_type"] != "classification":
        return

    if X_test_imp.shape[1] < 2:
        return

    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X_test_imp)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_test, palette="tab10", s=40)
    plt.title("Projection PCA - ensemble test")
    plt.tight_layout()
    plt.show()
