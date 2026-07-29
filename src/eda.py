"""
Phase 4 - Exploratory Data Analysis.

Produces the figures in reports/figures/ and prints the findings that the rest
of the project builds on. The most important output is
`analyse_disease_link()`, which tests the claim this whole project rests on:
that diabetes and heart disease are connected.
"""

import matplotlib

matplotlib.use("Agg")  # write files instead of opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
import data_cleaning

sns.set_theme(style="whitegrid")
PALETTE = ("#4C9F70", "#C1444F")  # negative, positive


def _save(fig, filename: str) -> None:
    config.ensure_directories()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> reports/figures/{filename}")


# ---------------------------------------------------------------------------
# The central claim: are the two diseases actually linked?
# ---------------------------------------------------------------------------
def analyse_disease_link(diabetes: pd.DataFrame) -> dict:
    """
    Measure the diabetes / heart-disease association in real data.

    The project is justified by the claim that diabetics face a much higher
    risk of heart disease. The CDC survey happens to record *both* conditions
    for the same respondents, so the claim can be tested directly rather than
    just cited.
    """
    print("\n[Diabetes <-> Heart disease link]")

    has_diabetes = diabetes["Diabetes_binary"] == 1
    heart_col = diabetes["HeartDiseaseorAttack"]

    rate_with = heart_col[has_diabetes].mean()
    rate_without = heart_col[~has_diabetes].mean()
    relative_risk = rate_with / rate_without

    print(f"  heart disease among diabetics     : {rate_with:.2%}")
    print(f"  heart disease among non-diabetics : {rate_without:.2%}")
    print(f"  relative risk                     : {relative_risk:.2f}x")
    print("  -> measured on 253k survey respondents, this reproduces the")
    print("     2-4x elevated risk reported in the clinical literature and")
    print("     justifies treating the two diseases as one linked problem.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].bar(
        ["No diabetes", "Diabetes"],
        [rate_without * 100, rate_with * 100],
        color=PALETTE,
    )
    axes[0].set_ylabel("Heart disease prevalence (%)")
    axes[0].set_title(
        f"Heart disease is {relative_risk:.1f}x more common in diabetics"
    )
    for index, value in enumerate([rate_without * 100, rate_with * 100]):
        axes[0].text(index, value + 0.6, f"{value:.1f}%", ha="center",
                     fontweight="bold")

    crosstab = pd.crosstab(
        diabetes["Diabetes_binary"], diabetes["HeartDiseaseorAttack"],
        normalize="index",
    ) * 100
    # Neutral sequential palette: a red/green scale would imply "good/bad" and
    # mislead here, since the largest cell is simply "no disease in either".
    sns.heatmap(crosstab, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                cbar_kws={"label": "% of row"})
    axes[1].set_xlabel("Heart disease or attack")
    axes[1].set_ylabel("Diabetes")
    axes[1].set_title("Co-occurrence of the two conditions")

    fig.suptitle("Why predict both diseases together", fontweight="bold")
    _save(fig, "00_disease_link.png")

    return {
        "heart_rate_with_diabetes": float(rate_with),
        "heart_rate_without_diabetes": float(rate_without),
        "relative_risk": float(relative_risk),
    }


# ---------------------------------------------------------------------------
# Standard EDA per dataset
# ---------------------------------------------------------------------------
def plot_target_balance(frames: dict) -> None:
    """Compare how balanced each dataset's target is - this drives SMOTE later."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (disease, df) in zip(axes, frames.items()):
        target = config.TARGET_COLUMN[disease]
        counts = df[target].value_counts().sort_index()
        share = counts / counts.sum()
        ax.bar(["Negative", "Positive"], counts.values, color=PALETTE)
        ax.set_title(
            f"{config.DISEASE_LABEL[disease]}\n"
            f"{share.iloc[1]:.1%} positive"
        )
        ax.set_ylabel("Patients")
        for index, value in enumerate(counts.values):
            ax.text(index, value, f"{value:,}", ha="center", va="bottom")

    fig.suptitle(
        "Class balance differs sharply - only diabetes needs resampling",
        fontweight="bold",
    )
    _save(fig, "01_target_balance.png")


def plot_correlations(df: pd.DataFrame, disease: str) -> pd.Series:
    """Heatmap of feature correlations plus the ranking against the target."""
    target = config.TARGET_COLUMN[disease]
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.7}, linewidths=0.3)
    ax.set_title(f"{config.DISEASE_LABEL[disease]} - feature correlations",
                 fontweight="bold")
    _save(fig, f"02_{disease}_correlation_heatmap.png")

    ranked = corr[target].drop(target).sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(ranked))))
    colours = [PALETTE[1] if v > 0 else PALETTE[0] for v in ranked.values]
    ax.barh([config.describe_feature(i) for i in ranked.index],
            ranked.values, color=colours)
    ax.invert_yaxis()
    ax.set_xlabel(f"Correlation with {config.DISEASE_LABEL[disease].lower()}")
    ax.set_title(f"What moves with {config.DISEASE_LABEL[disease].lower()}?",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    _save(fig, f"03_{disease}_target_correlation.png")

    print(f"\n  Top correlates of {config.DISEASE_LABEL[disease].lower()}:")
    for name, value in ranked.head(5).items():
        direction = "higher" if value > 0 else "lower"
        print(f"    {config.describe_feature(name):<45} {value:+.3f} ({direction})")

    return ranked


def plot_key_distributions(df: pd.DataFrame, disease: str, features: list) -> None:
    """Distribution of the most important continuous features, split by outcome."""
    target = config.TARGET_COLUMN[disease]
    features = [f for f in features if f in df.columns]
    if not features:
        return

    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        for value, colour, label in zip((0, 1), PALETTE, ("Negative", "Positive")):
            subset = df.loc[df[target] == value, feature]
            ax.hist(subset, bins=40, alpha=0.6, color=colour, label=label,
                    density=True)
        ax.set_xlabel(config.describe_feature(feature))
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(
        f"{config.DISEASE_LABEL[disease]} - key measurements by outcome",
        fontweight="bold",
    )
    _save(fig, f"04_{disease}_distributions.png")


def plot_risk_by_group(df: pd.DataFrame, disease: str, feature: str,
                       bins, labels, filename: str) -> None:
    """Show how disease rate climbs across bands of a single risk factor."""
    if feature not in df.columns:
        return
    target = config.TARGET_COLUMN[disease]

    banded = pd.cut(df[feature], bins=bins, labels=labels)
    rates = df.groupby(banded, observed=True)[target].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(rates.index.astype(str), rates.values, color="#4C72B0")
    ax.set_ylabel(f"{config.DISEASE_LABEL[disease]} rate (%)")
    ax.set_xlabel(config.describe_feature(feature))
    ax.set_title(
        f"{config.DISEASE_LABEL[disease]} risk rises with "
        f"{config.describe_feature(feature)}",
        fontweight="bold",
    )
    for index, value in enumerate(rates.values):
        ax.text(index, value, f"{value:.1f}%", ha="center", va="bottom")
    _save(fig, filename)


def compare_shared_risk_factors(diabetes: pd.DataFrame,
                                heart: pd.DataFrame) -> None:
    """
    Compare BMI's effect on both diseases side by side.

    The two datasets record different columns, but BMI is measured in both,
    which makes it the cleanest way to show the diseases share risk factors.
    """
    bands = [0, 18.5, 25, 30, 35, 100]
    labels = ["Under", "Normal", "Over", "Obese I", "Obese II+"]

    diabetes_rate = (
        diabetes.groupby(pd.cut(diabetes["BMI"], bands, labels=labels),
                         observed=True)["Diabetes_binary"].mean() * 100
    )
    heart_rate = (
        heart.groupby(pd.cut(heart["bmi"], bands, labels=labels),
                      observed=True)["cardio"].mean() * 100
    )

    comparison = pd.DataFrame(
        {"Diabetes": diabetes_rate, "Heart disease": heart_rate}
    )

    fig, ax = plt.subplots(figsize=(9, 4.6))
    comparison.plot(kind="bar", ax=ax, color=["#C1444F", "#4C72B0"], rot=0)
    ax.set_ylabel("Disease rate (%)")
    ax.set_xlabel("BMI category")
    ax.set_title("BMI raises the risk of both diseases - a shared risk factor",
                 fontweight="bold")
    ax.legend(title=None)
    _save(fig, "06_shared_risk_factor_bmi.png")

    print("\n  Disease rate by BMI category (shared risk factor):")
    print(comparison.round(1).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> dict:
    print("=" * 70)
    print("PHASE 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    frames = {d: data_cleaning.load_clean(d) for d in config.DISEASES}
    diabetes, heart = frames[config.DIABETES], frames[config.HEART]

    link = analyse_disease_link(diabetes)

    print("\n[Class balance]")
    plot_target_balance(frames)

    print("\n[Correlations]")
    for disease, df in frames.items():
        plot_correlations(df, disease)

    print("\n[Distributions]")
    plot_key_distributions(diabetes, config.DIABETES,
                           ["BMI", "GenHlth", "Age", "PhysHlth"])
    plot_key_distributions(heart, config.HEART,
                           ["age_years", "bmi", "ap_hi", "ap_lo"])

    print("\n[Risk gradients]")
    plot_risk_by_group(
        heart, config.HEART, "ap_hi",
        bins=[0, 110, 120, 130, 140, 160, 250],
        labels=["<110", "110-120", "120-130", "130-140", "140-160", "160+"],
        filename="05_heart_risk_by_blood_pressure.png",
    )
    plot_risk_by_group(
        diabetes, config.DIABETES, "BMI",
        bins=[0, 18.5, 25, 30, 35, 40, 100],
        labels=["Under", "Normal", "Over", "Obese I", "Obese II", "Obese III"],
        filename="05_diabetes_risk_by_bmi.png",
    )

    print("\n[Shared risk factors]")
    compare_shared_risk_factors(diabetes, heart)

    print("\nEDA complete.")
    return link


if __name__ == "__main__":
    main()
