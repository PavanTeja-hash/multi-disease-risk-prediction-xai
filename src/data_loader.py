"""
Phase 2 - Data Collection.

Loads the two raw datasets from disk and reports what was loaded. Nothing is
modified here; cleaning happens in the next phase. Keeping loading separate
from cleaning means the raw files always stay the untouched source of truth.
"""

import pandas as pd

import config


def load_diabetes() -> pd.DataFrame:
    """Load the CDC Diabetes Health Indicators dataset (BRFSS survey data)."""
    if not config.DIABETES_RAW.exists():
        raise FileNotFoundError(
            f"Diabetes dataset not found at {config.DIABETES_RAW}. "
            "Run scripts/download_data.py first."
        )
    return pd.read_csv(config.DIABETES_RAW)


def load_heart() -> pd.DataFrame:
    """Load the Cardiovascular Disease dataset (semicolon-separated)."""
    if not config.HEART_RAW.exists():
        raise FileNotFoundError(
            f"Heart dataset not found at {config.HEART_RAW}. "
            "Run scripts/download_data.py first."
        )
    return pd.read_csv(config.HEART_RAW, sep=config.HEART_SEPARATOR)


def load_raw(disease: str) -> pd.DataFrame:
    """Load whichever raw dataset corresponds to `disease`."""
    if disease == config.DIABETES:
        return load_diabetes()
    if disease == config.HEART:
        return load_heart()
    raise ValueError(f"Unknown disease key: {disease!r}")


def summarise(df: pd.DataFrame, name: str, target: str) -> None:
    """Print a short profile of a freshly loaded dataset."""
    positives = int(df[target].sum())
    total = len(df)
    print(f"\n{name}")
    print(f"  rows x columns : {df.shape[0]:,} x {df.shape[1]}")
    print(f"  missing values : {int(df.isna().sum().sum()):,}")
    print(f"  duplicate rows : {int(df.duplicated().sum()):,}")
    print(
        f"  target '{target}': {positives:,} positive "
        f"({positives / total:.1%}), {total - positives:,} negative"
    )


def main() -> None:
    print("=" * 70)
    print("PHASE 2: DATA COLLECTION")
    print("=" * 70)

    for disease in config.DISEASES:
        df = load_raw(disease)
        summarise(df, config.DISEASE_LABEL[disease], config.TARGET_COLUMN[disease])

    print(
        "\nBoth datasets loaded. They are kept separate on purpose: there is no "
        "public dataset labelling the same patients for both diseases, so each "
        "disease gets its own model trained on its own data."
    )


if __name__ == "__main__":
    main()
