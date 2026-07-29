"""
Phase 9 - Business Insights.

The models answer "who is at risk". This phase answers the questions a hospital
or insurer would actually ask next: which groups should we screen first, which
risk factors are worth an intervention programme, and what does it cost to run
this at a given sensitivity.

Everything here is computed from the cleaned data and the trained models, and
is written out to reports/ so it can be quoted directly.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data_cleaning
import explainability
import feature_engineering
import threshold_tuning
import train_models


def _save(fig, filename: str) -> None:
    config.ensure_directories()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


# ---------------------------------------------------------------------------
# Who is most at risk?
# ---------------------------------------------------------------------------
def highest_risk_segments(diabetes: pd.DataFrame, heart: pd.DataFrame) -> dict:
    """Find the patient groups with the highest observed disease rates."""
    print("\n[Highest-risk segments]")
    findings = {}

    # --- Diabetes: general health and BMI are the strongest survey signals ---
    diabetes_segments = (
        diabetes.groupby(["GenHlth", "bmi_band"], observed=True)["Diabetes_binary"]
        .agg(["mean", "size"])
        .reset_index()
    )
    # Ignore tiny groups; a 100% rate over 12 people is noise, not a finding.
    diabetes_segments = diabetes_segments[diabetes_segments["size"] >= 500]
    worst_diabetes = diabetes_segments.nlargest(5, "mean")

    print("\n  Diabetes - highest-rate groups (min. 500 people):")
    for row in worst_diabetes.itertuples():
        print(f"    general health {row.GenHlth}/5, BMI {row.bmi_band:<10} "
              f"-> {row.mean:.1%} diabetic  ({row.size:,} people)")
    findings["diabetes_segments"] = worst_diabetes

    # --- Heart: blood pressure and age dominate ---
    heart_segments = (
        heart.groupby(["bp_band", "age_band"], observed=True)["cardio"]
        .agg(["mean", "size"])
        .reset_index()
    )
    heart_segments = heart_segments[heart_segments["size"] >= 500]
    worst_heart = heart_segments.nlargest(5, "mean")

    print("\n  Heart disease - highest-rate groups (min. 500 people):")
    for row in worst_heart.itertuples():
        print(f"    blood pressure {row.bp_band:<12} age {row.age_band:<8} "
              f"-> {row.mean:.1%} affected  ({row.size:,} people)")
    findings["heart_segments"] = worst_heart

    return findings


def plot_risk_matrix(diabetes: pd.DataFrame, heart: pd.DataFrame) -> None:
    """Two heatmaps showing where risk concentrates for each disease."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    diabetes_matrix = diabetes.pivot_table(
        index="GenHlth", columns="bmi_band", values="Diabetes_binary",
        aggfunc="mean", observed=True,
    ) * 100
    im0 = axes[0].imshow(diabetes_matrix, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(len(diabetes_matrix.columns)),
                       diabetes_matrix.columns, rotation=30)
    axes[0].set_yticks(range(len(diabetes_matrix.index)), diabetes_matrix.index)
    axes[0].set_xlabel("BMI band")
    axes[0].set_ylabel("Self-reported health (1=excellent, 5=poor)")
    axes[0].set_title("Diabetes rate (%)", fontweight="bold")
    for i in range(diabetes_matrix.shape[0]):
        for j in range(diabetes_matrix.shape[1]):
            value = diabetes_matrix.iloc[i, j]
            if not np.isnan(value):
                axes[0].text(j, i, f"{value:.0f}", ha="center", va="center",
                             fontsize=8,
                             color="white" if value > 30 else "black")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    heart_matrix = heart.pivot_table(
        index="bp_band", columns="age_band", values="cardio",
        aggfunc="mean", observed=True,
    ) * 100
    im1 = axes[1].imshow(heart_matrix, cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(len(heart_matrix.columns)), heart_matrix.columns)
    axes[1].set_yticks(range(len(heart_matrix.index)), heart_matrix.index)
    axes[1].set_xlabel("Age band")
    axes[1].set_ylabel("Blood pressure category")
    axes[1].set_title("Heart disease rate (%)", fontweight="bold")
    for i in range(heart_matrix.shape[0]):
        for j in range(heart_matrix.shape[1]):
            value = heart_matrix.iloc[i, j]
            if not np.isnan(value):
                axes[1].text(j, i, f"{value:.0f}", ha="center", va="center",
                             fontsize=8,
                             color="white" if value > 50 else "black")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Where risk concentrates", fontweight="bold")
    fig.tight_layout()
    _save(fig, "14_risk_matrix.png")


# ---------------------------------------------------------------------------
# Shared vs distinct drivers
# ---------------------------------------------------------------------------
def compare_drivers(datasets: dict) -> pd.DataFrame:
    """
    Compare what drives each disease, using SHAP importance from each model.

    This is the part that only a multi-disease project can produce: a single
    view of which factors matter for both conditions and which are specific to
    one.
    """
    print("\n[What drives each disease]")
    rankings = {}

    for disease in config.DISEASES:
        dataset = datasets[disease]
        model = train_models.load(disease, train_models.GRADIENT_BOOSTING)
        shap_values, _, _, _ = explainability.compute_shap(
            model, dataset, sample_size=1500
        )
        importance = explainability.global_importance(shap_values, dataset)
        # Normalise so the two diseases are comparable despite different scales.
        importance = importance / importance.sum()

        # Collapse onto shared clinical concepts *before* combining the two
        # models. Doing it afterwards would leave "BMI" and "bmi" as separate
        # rows, each zero for the other disease, and the shared-factor analysis
        # would find nothing.
        importance.index = [config.canonical_concept(f) for f in importance.index]
        rankings[disease] = importance.groupby(level=0).sum()

    comparison = pd.DataFrame(rankings).fillna(0)
    comparison.columns = [config.DISEASE_LABEL[d] for d in comparison.columns]

    shared = comparison[(comparison > 0.02).all(axis=1)]
    shared = shared.assign(total=shared.sum(axis=1)).sort_values(
        "total", ascending=False).drop(columns="total")

    print("\n  Factors that matter for BOTH diseases:")
    if shared.empty:
        print("    (none above the 2% importance threshold in both models)")
    else:
        for name, row in shared.iterrows():
            print(f"    {name:<45} "
                  f"diabetes {row.iloc[0]:.1%} | heart {row.iloc[1]:.1%}")

    print("\n  Top driver unique to each disease:")
    for column in comparison.columns:
        other = [c for c in comparison.columns if c != column][0]
        unique = comparison[(comparison[column] > 0.05)
                            & (comparison[other] < 0.01)]
        if not unique.empty:
            top = unique[column].idxmax()
            print(f"    {column:<15} {top} ({unique.loc[top, column]:.1%})")

    fig, ax = plt.subplots(figsize=(9, 7))
    top_features = comparison.assign(
        total=comparison.sum(axis=1)).nlargest(12, "total").drop(columns="total")
    top_features.plot(kind="barh", ax=ax, color=["#C1444F", "#4C72B0"])
    ax.set_xlabel("Share of the model's total explanatory power")
    ax.set_title("Shared and disease-specific risk factors", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(title=None)
    _save(fig, "15_shared_vs_specific_drivers.png")

    return comparison


# ---------------------------------------------------------------------------
# What screening at this sensitivity actually costs
# ---------------------------------------------------------------------------
def screening_workload(datasets: dict) -> pd.DataFrame:
    """
    Translate the tuned threshold into operational numbers.

    A recall target is a clinical decision, but it has a cost: the higher the
    recall, the more healthy people get flagged for follow-up. This makes that
    trade explicit instead of leaving it inside an F1 score.
    """
    print("\n[What screening actually costs]")
    rows = []

    for disease in config.DISEASES:
        dataset = datasets[disease]
        model = train_models.load(disease, train_models.GRADIENT_BOOSTING)

        val_probabilities = model.predict_proba(dataset.X_val)[:, 1]
        test_probabilities = model.predict_proba(dataset.X_test)[:, 1]

        for target in (0.70, 0.80, 0.90):
            threshold = threshold_tuning.threshold_for_recall(
                dataset.y_val, val_probabilities, target
            )
            scores = threshold_tuning.evaluate_at(
                dataset.y_test, test_probabilities, threshold
            )
            flagged = int((test_probabilities >= threshold).sum())
            total = len(dataset.y_test)
            true_cases = int(dataset.y_test.sum())

            rows.append({
                "Disease": config.DISEASE_LABEL[disease],
                "Recall target": f"{target:.0%}",
                "Threshold": round(threshold, 3),
                "Recall achieved": round(scores["recall"], 3),
                "Precision": round(scores["precision"], 3),
                "Flagged per 1000": round(1000 * flagged / total),
                "Cases found": int(scores["recall"] * true_cases),
                "Cases missed": true_cases - int(scores["recall"] * true_cases),
            })

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    config.ensure_directories()
    path = config.REPORTS_DIR / "screening_workload.csv"
    table.to_csv(path, index=False)
    print(f"\n    table -> reports/{path.name}")
    return table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def add_bands(diabetes: pd.DataFrame, heart: pd.DataFrame):
    """Add readable bands used for the segment analysis."""
    diabetes = diabetes.copy()
    diabetes["bmi_band"] = pd.cut(
        diabetes["BMI"], [0, 25, 30, 35, 100],
        labels=["normal", "overweight", "obese I", "obese II+"],
    )

    heart = heart.copy()
    heart["bp_band"] = pd.cut(
        heart["ap_hi"], [0, 120, 130, 140, 250],
        labels=["normal", "elevated", "stage 1", "stage 2+"],
    )
    heart["age_band"] = pd.cut(
        heart["age_years"], [0, 45, 50, 55, 60, 120],
        labels=["<45", "45-50", "50-55", "55-60", "60+"],
    )
    return diabetes, heart


def main() -> None:
    print("=" * 70)
    print("PHASE 9: BUSINESS INSIGHTS")
    print("=" * 70)

    diabetes = data_cleaning.load_clean(config.DIABETES)
    heart = data_cleaning.load_clean(config.HEART)
    diabetes, heart = add_bands(diabetes, heart)

    highest_risk_segments(diabetes, heart)
    plot_risk_matrix(diabetes, heart)

    datasets = feature_engineering.prepare_all(verbose=False)
    compare_drivers(datasets)
    screening_workload(datasets)

    print("\nInsights complete.")


if __name__ == "__main__":
    main()
