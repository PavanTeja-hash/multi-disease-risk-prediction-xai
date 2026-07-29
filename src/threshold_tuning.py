"""
Phase 7b - Decision threshold tuning.

A classifier outputs a probability. Turning that into a yes/no answer needs a
cut-off, and 0.5 is only the right cut-off by accident.

Phase 7 showed exactly why this matters: on the diabetes data, Gradient
Boosting had the best ROC-AUC of any model - it ranks sick patients above
healthy ones better than anything else - yet at a 0.5 threshold it caught only
about a fifth of real diabetics. The ranking was good; the cut-off was wrong.

Two rules are followed here, both of which are easy to get wrong:

1. The threshold is chosen on the *validation* set, never the test set.
   Choosing it on test data and then reporting test scores means reporting a
   number that was fitted to the very data it claims to be measured on.

2. The validation set keeps its natural class balance (roughly 14% diabetic),
   not the SMOTE-balanced 50/50 of the training data. A threshold tuned on
   resampled data is calibrated for a population that does not exist.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import config


def sweep(y_true: np.ndarray, probabilities: np.ndarray,
          steps: int = 200) -> dict:
    """Score every candidate threshold between 0 and 1."""
    thresholds = np.linspace(0.01, 0.99, steps)
    rows = {"threshold": thresholds, "precision": [], "recall": [], "f1": []}

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows["precision"].append(
            precision_score(y_true, predictions, zero_division=0))
        rows["recall"].append(
            recall_score(y_true, predictions, zero_division=0))
        rows["f1"].append(f1_score(y_true, predictions, zero_division=0))

    for key in ("precision", "recall", "f1"):
        rows[key] = np.array(rows[key])
    return rows


def best_f1_threshold(y_val: np.ndarray, probabilities: np.ndarray) -> float:
    """The threshold giving the best precision/recall balance on validation."""
    curve = sweep(y_val, probabilities)
    return float(curve["threshold"][int(np.argmax(curve["f1"]))])


def threshold_for_recall(y_val: np.ndarray, probabilities: np.ndarray,
                         target_recall: float = 0.80) -> float:
    """
    The highest threshold that still reaches a target recall.

    Screening tools are usually specified this way: "catch at least 80% of
    cases", then keep precision as high as that allows.
    """
    curve = sweep(y_val, probabilities)
    feasible = curve["threshold"][curve["recall"] >= target_recall]
    if len(feasible) == 0:
        return float(curve["threshold"][int(np.argmax(curve["recall"]))])
    return float(feasible.max())


def evaluate_at(y_true: np.ndarray, probabilities: np.ndarray,
                threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def plot_threshold_curve(y_val: np.ndarray, probabilities: np.ndarray,
                         chosen: float, disease: str, model_name: str) -> None:
    curve = sweep(y_val, probabilities)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(curve["threshold"], curve["precision"], label="Precision",
            linewidth=2)
    ax.plot(curve["threshold"], curve["recall"], label="Recall", linewidth=2)
    ax.plot(curve["threshold"], curve["f1"], label="F1", linewidth=2)
    ax.axvline(0.5, color="grey", linestyle=":", label="Default 0.5")
    ax.axvline(chosen, color="crimson", linestyle="--",
               label=f"Chosen {chosen:.2f}")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"{config.DISEASE_LABEL[disease]} - {model_name}\n"
        "threshold chosen on validation data, not test",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    config.ensure_directories()
    filename = f"10_{disease}_threshold_tuning.png"
    fig.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


def tune(model, dataset, model_name: str, target_recall: float = 0.80) -> dict:
    """Pick a threshold on validation data and report its effect on test data."""
    val_probabilities = model.predict_proba(dataset.X_val)[:, 1]
    test_probabilities = model.predict_proba(dataset.X_test)[:, 1]

    chosen = threshold_for_recall(dataset.y_val, val_probabilities,
                                  target_recall)

    default_test = evaluate_at(dataset.y_test, test_probabilities, 0.5)
    tuned_test = evaluate_at(dataset.y_test, test_probabilities, chosen)

    print(f"\n  {model_name} on {dataset.label}")
    print(f"    chosen threshold : {chosen:.3f} "
          f"(targeting {target_recall:.0%} recall on validation)")
    print(f"    default 0.50 -> recall {default_test['recall']:.3f}, "
          f"precision {default_test['precision']:.3f}, F1 {default_test['f1']:.3f}")
    print(f"    tuned  {chosen:.2f} -> recall {tuned_test['recall']:.3f}, "
          f"precision {tuned_test['precision']:.3f}, F1 {tuned_test['f1']:.3f}")

    caught_before = int(default_test["recall"] * dataset.y_test.sum())
    caught_after = int(tuned_test["recall"] * dataset.y_test.sum())
    print(f"    real patients found: {caught_before:,} -> {caught_after:,} "
          f"of {int(dataset.y_test.sum()):,}")

    plot_threshold_curve(dataset.y_val, val_probabilities, chosen,
                         dataset.disease, model_name)

    return {"threshold": chosen, "default": default_test, "tuned": tuned_test}
