"""
Phase 7 - Model Evaluation.

Scores every model on the held-out test set, which no model has seen at any
point during training.

Accuracy alone is deliberately not the headline. On the diabetes data 86% of
patients are healthy, so a model that predicts "no diabetes" for everybody
scores 86% accuracy while finding zero actual patients. The metrics that
matter here are:

* Recall    - of the people who really have the disease, how many were caught?
              A missed case is the expensive error in screening.
* Precision - of those flagged, how many really have it?
* F1        - the balance between the two.
* ROC-AUC   - how well the model ranks sick above healthy across every possible
              threshold. Unlike accuracy, it is not fooled by class imbalance.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
import feature_engineering
import threshold_tuning
import train_models
import train_neural_network


def score_model(model, X_test, y_test) -> dict:
    """Compute the full metric set for one fitted model."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "_probabilities": probabilities,
        "_predictions": predictions,
    }


def baseline_accuracy(y_test: np.ndarray) -> float:
    """Accuracy of always predicting the majority class - the bar to clear."""
    majority = 1 - y_test.mean() if y_test.mean() < 0.5 else y_test.mean()
    return float(majority)


def collect_models(disease: str) -> dict:
    """Load every trained model for one disease behind a common interface."""
    models = dict(train_models.load_all(disease))
    try:
        models[train_neural_network.NEURAL_NETWORK] = (
            train_neural_network.load_wrapped(disease)
        )
    except (FileNotFoundError, ImportError, OSError) as error:
        print(f"  (neural network unavailable: {type(error).__name__})")
    return models


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrices(results: dict, dataset, filename: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, scores) in zip(axes, results.items()):
        matrix = confusion_matrix(dataset.y_test, scores["_predictions"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred. healthy", "Pred. disease"])
        ax.set_yticks([0, 1], ["Healthy", "Disease"])
        threshold = matrix.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "black",
                        fontweight="bold")
        missed = matrix[1, 0]
        ax.set_title(f"{name}\n{missed:,} missed cases", fontsize=10)

    fig.suptitle(f"{dataset.label} - confusion matrices (test set)",
                 fontweight="bold")
    fig.tight_layout()
    config.ensure_directories()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


def plot_roc_curves(results: dict, dataset, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    for name, scores in results.items():
        fpr, tpr, _ = roc_curve(dataset.y_test, scores["_probabilities"])
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{name} (AUC {scores['ROC-AUC']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"{dataset.label} - ROC curves", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    config.ensure_directories()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


def plot_model_comparison(tables: dict, filename: str) -> None:
    """Bar chart comparing every model on both diseases."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, axes = plt.subplots(1, len(tables), figsize=(7.5 * len(tables), 4.8))
    if len(tables) == 1:
        axes = [axes]

    for ax, (disease, table) in zip(axes, tables.items()):
        table[metrics].plot(kind="bar", ax=ax, rot=15, width=0.8,
                            colormap="viridis")
        ax.set_title(config.DISEASE_LABEL[disease], fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model comparison across both diseases", fontweight="bold")
    fig.tight_layout()
    config.ensure_directories()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def evaluate_disease(dataset) -> pd.DataFrame:
    print(f"\n{dataset.label}")
    models = collect_models(dataset.disease)

    results = {
        name: score_model(model, dataset.X_test, dataset.y_test)
        for name, model in models.items()
    }

    table = pd.DataFrame(
        {name: {k: v for k, v in scores.items() if not k.startswith("_")}
         for name, scores in results.items()}
    ).T

    naive = baseline_accuracy(dataset.y_test)
    print(f"  always-predict-majority accuracy: {naive:.3f}  "
          "<- the number accuracy alone would have to beat")
    print()
    print(table.round(4).to_string())

    best = table["ROC-AUC"].idxmax()
    print(f"\n  best by ROC-AUC : {best} ({table.loc[best, 'ROC-AUC']:.4f})")
    print(f"  best by recall  : {table['Recall'].idxmax()} "
          f"({table['Recall'].max():.4f})")

    plot_confusion_matrices(results, dataset,
                            f"07_{dataset.disease}_confusion_matrices.png")
    plot_roc_curves(results, dataset, f"08_{dataset.disease}_roc_curves.png")

    # The model that ranks patients best is the one worth deploying, but its
    # default cut-off may be badly suited to screening. Tune it properly.
    print(f"\n  [threshold tuning - {best}]")
    tuning = threshold_tuning.tune(models[best], dataset, best)
    table.attrs["best_model"] = best
    table.attrs["tuning"] = tuning

    return table


def main() -> dict:
    print("=" * 70)
    print("PHASE 7: MODEL EVALUATION")
    print("=" * 70)

    datasets = feature_engineering.prepare_all(verbose=False)
    tables = {d: evaluate_disease(datasets[d]) for d in config.DISEASES}

    plot_model_comparison(tables, "09_model_comparison.png")

    config.ensure_directories()
    for disease, table in tables.items():
        path = config.REPORTS_DIR / f"{disease}_model_scores.csv"
        table.round(4).to_csv(path)
        print(f"    scores -> reports/{path.name}")

    print("\nEvaluation complete.")
    return tables


if __name__ == "__main__":
    main()
