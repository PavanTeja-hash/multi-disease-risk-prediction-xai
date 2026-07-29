"""
Phase 8 - Explainable AI (SHAP and LIME).

A risk score on its own is not clinically useful. "You have a 73% chance of
diabetes" invites the obvious question - why? - and a model that cannot answer
it will not be trusted or acted on. This phase opens the box.

Two complementary techniques:

* SHAP  - built on Shapley values from cooperative game theory. It treats each
          feature as a player in a game whose payout is the prediction, and
          fairly divides credit for the outcome between them. Its strength is
          consistency: the contributions always add up exactly to the
          prediction, and the same method gives both per-patient and
          whole-population views.

* LIME  - approximates the complex model with a simple linear one in the small
          neighbourhood around a single patient. It answers a slightly
          different question ("what does the model do *near this person*") and
          is used here as a second opinion. When two methods built on different
          assumptions agree, the explanation is more credible.

SHAP values are returned in log-odds, not percentage points. A value of +0.8
means that feature pushed the log-odds of disease up by 0.8 - it does not mean
"+80% risk". Reporting them as percentages would be wrong, and the report
generator in Phase 8b is careful about this.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import feature_engineering
import train_models

# Explaining every one of 50,000 test patients is unnecessary and slow; a
# random sample is enough to characterise global behaviour.
GLOBAL_SAMPLE_SIZE = 2000
LIME_TRAINING_SAMPLE = 5000


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def build_shap_explainer(model, X_background: np.ndarray):
    """
    Create the most appropriate SHAP explainer for the given model.

    Tree-based models have an exact, fast algorithm (TreeExplainer). Anything
    else falls back to a model-agnostic explainer, which is slower but works on
    any predict_proba function - including the Keras network.
    """
    import shap

    try:
        return shap.TreeExplainer(model), "TreeExplainer (exact, fast)"
    except Exception:
        summary = shap.kmeans(X_background, 25)
        return (
            shap.KernelExplainer(lambda data: model.predict_proba(data)[:, 1],
                                 summary),
            "KernelExplainer (model-agnostic fallback)",
        )


def _positive_class_values(shap_values) -> np.ndarray:
    """
    Normalise SHAP output to a plain (n_samples, n_features) array.

    Different model types and SHAP versions return different shapes for binary
    classification - sometimes a list of two arrays, sometimes a 3D array with
    a trailing class axis. This flattens all of them to the positive class.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def compute_shap(model, dataset, sample_size: int = GLOBAL_SAMPLE_SIZE):
    """Compute SHAP values for a sample of the test set."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    size = min(sample_size, len(dataset.X_test))
    index = rng.choice(len(dataset.X_test), size=size, replace=False)

    X_sample = dataset.X_test[index]
    explainer, kind = build_shap_explainer(model, dataset.X_train[:500])
    print(f"    explainer: {kind}")

    values = _positive_class_values(explainer.shap_values(X_sample))
    return values, X_sample, index, explainer


def plot_shap_summary(shap_values: np.ndarray, X_sample: np.ndarray,
                      dataset, filename: str) -> None:
    """Beeswarm plot: which features matter, and in which direction."""
    import shap

    readable = [config.describe_feature(f) for f in dataset.feature_names]

    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_sample, columns=readable),
        show=False,
        plot_size=None,
    )
    plt.title(f"{dataset.label} - what drives the model's predictions",
              fontweight="bold")
    config.ensure_directories()
    plt.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    figure -> reports/figures/{filename}")


def global_importance(shap_values: np.ndarray, dataset) -> pd.Series:
    """Rank features by mean absolute SHAP value."""
    importance = np.abs(shap_values).mean(axis=0)
    series = pd.Series(importance, index=dataset.feature_names)
    return series.sort_values(ascending=False)


def plot_global_importance(importance: pd.Series, dataset,
                           filename: str, top_n: int = 15) -> None:
    top = importance.head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(top) + 1.6))
    ax.barh([config.describe_feature(i) for i in top.index], top.values,
            color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|  (average impact on the prediction)")
    ax.set_title(f"{dataset.label} - most influential factors overall",
                 fontweight="bold")

    config.ensure_directories()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


# ---------------------------------------------------------------------------
# Per-patient explanation
# ---------------------------------------------------------------------------
def explain_patient(model, dataset, patient_position: int,
                    explainer=None, top_n: int = 6) -> dict:
    """
    Explain a single prediction.

    Returns the raw material for a plain-English report: the risk score, and
    the factors that pushed it up or down, described in real units rather than
    standardised ones.
    """
    import shap

    X_row = dataset.X_test[patient_position: patient_position + 1]
    raw_row = dataset.X_test_raw.iloc[patient_position]

    probability = float(model.predict_proba(X_row)[0, 1])

    if explainer is None:
        explainer, _ = build_shap_explainer(model, dataset.X_train[:500])
    contributions = _positive_class_values(explainer.shap_values(X_row))[0]

    frame = pd.DataFrame({
        "feature": dataset.feature_names,
        "value": [raw_row[f] for f in dataset.feature_names],
        "shap": contributions,
    })
    frame["magnitude"] = frame["shap"].abs()
    frame = frame.sort_values("magnitude", ascending=False)

    top = frame.head(top_n)
    return {
        "disease": dataset.disease,
        "probability": probability,
        "actual": int(dataset.y_test[patient_position]),
        "increasing": [
            {"feature": r.feature, "value": r.value, "shap": float(r.shap)}
            for r in top.itertuples() if r.shap > 0
        ],
        "decreasing": [
            {"feature": r.feature, "value": r.value, "shap": float(r.shap)}
            for r in top.itertuples() if r.shap < 0
        ],
        "all_contributions": frame,
    }


def plot_patient_waterfall(explanation: dict, dataset, filename: str) -> None:
    """Show how one patient's factors add up to their score."""
    frame = explanation["all_contributions"].head(10).iloc[::-1]
    colours = ["#C1444F" if v > 0 else "#4C9F70" for v in frame["shap"]]
    labels = [
        f"{config.describe_feature(f)} = {v:g}"
        for f, v in zip(frame["feature"], frame["value"])
    ]

    fig, ax = plt.subplots(figsize=(9.5, 0.46 * len(frame) + 1.8))
    ax.barh(labels, frame["shap"], color=colours)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("SHAP contribution (log-odds)  -  right = raises risk")
    ax.set_title(
        f"{dataset.label}: why this patient scored "
        f"{explanation['probability']:.0%}",
        fontweight="bold",
    )

    config.ensure_directories()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


# ---------------------------------------------------------------------------
# LIME - the independent second opinion
# ---------------------------------------------------------------------------
# LIME builds a simple linear model in a neighbourhood around the patient, and
# the size of that neighbourhood is a free parameter. Too wide and the local
# model is not local; too narrow and there is no variation left to learn from.
# Both failures show up as a low R^2, so the width is chosen by measuring
# fidelity rather than guessing.
LIME_KERNEL_WIDTHS = (1.0, 2.0, 3.0, 4.0)


def explain_patient_with_lime(model, dataset, patient_position: int,
                              top_n: int = 6) -> dict:
    """
    Explain the same patient with LIME, to cross-check SHAP.

    Two details matter here, and getting either wrong produces an explanation
    that looks fine but means nothing:

    * The background sample comes from the validation set, not the training
      set. The training set has been SMOTE-balanced to 50/50, so perturbing
      within it explores a population that does not exist and the model
      saturates near certainty - which drove local R^2 to 0.005 in testing.

    * The neighbourhood width is selected by local R^2. With 25 standardised
      features LIME's default width is far too wide to be "local" at all.
    """
    from lime.lime_tabular import LimeTabularExplainer

    rng = np.random.RandomState(config.RANDOM_STATE)
    source = dataset.X_val if len(dataset.X_val) else dataset.X_train
    size = min(LIME_TRAINING_SAMPLE, len(source))
    background = source[rng.choice(len(source), size, replace=False)]

    readable = [config.describe_feature(f) for f in dataset.feature_names]

    best = None
    for width in LIME_KERNEL_WIDTHS:
        explainer = LimeTabularExplainer(
            background,
            feature_names=readable,
            class_names=["healthy", config.DISEASE_LABEL[dataset.disease]],
            discretize_continuous=False,
            kernel_width=width,
            random_state=config.RANDOM_STATE,
        )
        explanation = explainer.explain_instance(
            dataset.X_test[patient_position],
            lambda data: model.predict_proba(data),
            num_features=top_n,
        )
        if best is None or explanation.score > best["r2"]:
            best = {
                "factors": explanation.as_list(),
                "r2": float(explanation.score),
                "kernel_width": width,
            }

    return best


def compare_shap_and_lime(shap_explanation: dict, lime_output: dict,
                          dataset) -> None:
    """
    Print SHAP and LIME side by side for the same patient.

    Only the *direction and ranking* of the factors should be compared. The
    magnitudes are not on the same scale: SHAP values here are log-odds, while
    LIME weights are coefficients of a local linear fit in probability space.
    """
    print("\n    SHAP - top factors raising risk:")
    for item in shap_explanation["increasing"][:4]:
        print(f"      {config.describe_feature(item['feature']):<45} "
              f"= {item['value']:<8g} (+{item['shap']:.3f})")
    if shap_explanation["decreasing"]:
        print("    SHAP - top factors lowering risk:")
        for item in shap_explanation["decreasing"][:3]:
            print(f"      {config.describe_feature(item['feature']):<45} "
                  f"= {item['value']:<8g} ({item['shap']:.3f})")

    print(f"\n    LIME - independent check "
          f"(local fit R^2 = {lime_output['r2']:.3f}, "
          f"neighbourhood width {lime_output['kernel_width']}):")
    for description, weight in lime_output["factors"][:4]:
        direction = "raises" if weight > 0 else "lowers"
        print(f"      {description:<45} {direction} risk ({weight:+.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def pick_high_risk_patient(model, dataset) -> int:
    """Find a genuinely high-risk patient - a more interesting example."""
    probabilities = model.predict_proba(dataset.X_test[:3000])[:, 1]
    return int(np.argmax(probabilities))


def main() -> dict:
    print("=" * 70)
    print("PHASE 8: EXPLAINABLE AI (SHAP + LIME)")
    print("=" * 70)

    datasets = feature_engineering.prepare_all(verbose=False)
    output = {}

    for disease in config.DISEASES:
        dataset = datasets[disease]
        model = train_models.load(disease, train_models.GRADIENT_BOOSTING)
        print(f"\n{dataset.label}")

        shap_values, X_sample, _, explainer = compute_shap(model, dataset)

        plot_shap_summary(shap_values, X_sample, dataset,
                          f"11_{disease}_shap_summary.png")
        importance = global_importance(shap_values, dataset)
        plot_global_importance(importance, dataset,
                               f"12_{disease}_shap_importance.png")

        print("\n    Most influential factors overall:")
        for name, value in importance.head(5).items():
            print(f"      {config.describe_feature(name):<45} {value:.4f}")

        position = pick_high_risk_patient(model, dataset)
        explanation = explain_patient(model, dataset, position,
                                      explainer=explainer)
        print(f"\n    Example patient #{position} - model says "
              f"{explanation['probability']:.1%} risk "
              f"(actual: {'has' if explanation['actual'] else 'does not have'} "
              f"{config.DISEASE_LABEL[disease].lower()})")

        plot_patient_waterfall(explanation, dataset,
                               f"13_{disease}_patient_explanation.png")

        lime_output = explain_patient_with_lime(model, dataset, position)
        compare_shap_and_lime(explanation, lime_output, dataset)

        output[disease] = {
            "importance": importance,
            "example_explanation": explanation,
            "lime": lime_output,
        }

    print("\nExplainability complete.")
    return output


if __name__ == "__main__":
    main()
