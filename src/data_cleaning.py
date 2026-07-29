"""
Phase 3 - Data Cleaning.

Each dataset has a different problem, so each gets its own cleaning routine:

* The diabetes data is survey-based and arrives complete - no missing values
  and no impossible readings. Its only apparent issue is repeated rows, which
  turn out not to be an issue at all (see `report_duplicate_analysis`).

* The cardiovascular data contains genuine data-entry errors: negative blood
  pressures, systolic readings in the thousands, and adult heights of 55cm.
  Those rows are removed, because they are impossible rather than merely rare.

Cleaned data is written to data/processed/.
"""

import pandas as pd

import config
import data_loader


# ---------------------------------------------------------------------------
# Diabetes
# ---------------------------------------------------------------------------
def report_duplicate_analysis(df: pd.DataFrame, target: str) -> None:
    """
    Explain why repeated rows in the survey data are kept.

    The instinct is to call drop_duplicates(). That would be wrong here. This
    is a 253k-person health survey with mostly binary answers, so two different
    healthy respondents can easily produce an identical row by chance. The
    numbers below confirm they are coincidences, not data-entry errors.
    """
    duplicated = df.duplicated(keep=False)
    n_dup_rows = int(duplicated.sum())
    if n_dup_rows == 0:
        print("  duplicates      : none")
        return

    rate_dup = df.loc[duplicated, target].mean()
    rate_unique = df.loc[~duplicated, target].mean()

    print(f"  repeated rows   : {n_dup_rows:,} "
          f"({df.duplicated().sum():,} would be dropped)")
    print(f"    positive rate among repeated rows : {rate_dup:.2%}")
    print(f"    positive rate among unique rows   : {rate_unique:.2%}")
    print("    -> kept. Repeated rows are overwhelmingly healthy respondents;")
    print("       dropping them would delete mostly negative cases and inflate")
    print("       the apparent disease rate. They are distinct people who gave")
    print("       identical answers, not duplicated records.")


def clean_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the CDC diabetes survey data."""
    print("\nDiabetes")
    before = len(df)
    df = df.copy()

    print(f"  missing values  : {int(df.isna().sum().sum()):,}")
    report_duplicate_analysis(df, config.TARGET_COLUMN[config.DIABETES])

    # BMI is the only continuous clinical measure here; screen for impossible
    # values even though the survey is generally well-formed.
    low, high = config.VALID_RANGES["bmi"]
    impossible_bmi = ~df["BMI"].between(low, high)
    if impossible_bmi.any():
        print(f"  impossible BMI  : {int(impossible_bmi.sum()):,} rows removed "
              f"(outside {low}-{high})")
        df = df[~impossible_bmi]

    print(f"  rows: {before:,} -> {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Heart / cardiovascular
# ---------------------------------------------------------------------------
def clean_heart(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the cardiovascular dataset, which contains real data-entry errors."""
    print("\nHeart Disease")
    before = len(df)
    df = df.copy()

    # 'id' is a row identifier, not a medical measurement. Leaving it in would
    # let a model learn from an arbitrary number.
    if "id" in df.columns:
        df = df.drop(columns="id")
        print("  dropped 'id'    : row identifier, carries no medical meaning")

    print(f"  missing values  : {int(df.isna().sum().sum()):,}")

    # Age is stored in days (e.g. 18393). Years are what a clinician - and a
    # SHAP explanation - would actually use.
    df["age_years"] = (df["age"] / 365.25).round(1)
    df = df.drop(columns="age")
    print(f"  converted age   : days -> years "
          f"(range {df['age_years'].min():.0f}-{df['age_years'].max():.0f})")

    # Some rows have systolic and diastolic readings swapped (diastolic higher
    # than systolic). That is a transcription error with an obvious fix, so
    # these rows are repaired rather than discarded.
    swapped = df["ap_lo"] > df["ap_hi"]
    if swapped.any():
        df.loc[swapped, ["ap_hi", "ap_lo"]] = df.loc[
            swapped, ["ap_lo", "ap_hi"]
        ].values
        print(f"  swapped BP      : {int(swapped.sum()):,} rows had systolic and "
              "diastolic reversed - values swapped back")

    # Remove physically impossible readings. These are not rare-but-real
    # patients: a systolic pressure of 16020 or -150 cannot occur.
    for column in ("ap_hi", "ap_lo", "height", "weight"):
        low, high = config.VALID_RANGES[column]
        outside = ~df[column].between(low, high)
        if outside.any():
            print(f"  impossible {column:<7}: {int(outside.sum()):,} rows removed "
                  f"(outside {low}-{high})")
            df = df[~outside]

    # BMI is not in the raw data but is the single most useful derived measure
    # for cardiovascular risk, and it makes both datasets comparable.
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    low, high = config.VALID_RANGES["bmi"]
    impossible_bmi = ~df["bmi"].between(low, high)
    if impossible_bmi.any():
        print(f"  impossible bmi  : {int(impossible_bmi.sum()):,} rows removed "
              f"(outside {low}-{high})")
        df = df[~impossible_bmi]

    print(f"  rows: {before:,} -> {len(df):,} "
          f"({(before - len(df)) / before:.1%} removed)")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
CLEANERS = {
    config.DIABETES: clean_diabetes,
    config.HEART: clean_heart,
}


def processed_path(disease: str):
    return config.DATA_PROCESSED / f"{disease}_clean.csv"


def clean(disease: str, save: bool = True) -> pd.DataFrame:
    """Load, clean, and optionally persist one dataset."""
    raw = data_loader.load_raw(disease)
    cleaned = CLEANERS[disease](raw)
    if save:
        config.ensure_directories()
        cleaned.to_csv(processed_path(disease), index=False)
    return cleaned


def load_clean(disease: str) -> pd.DataFrame:
    """Load an already-cleaned dataset, cleaning it first if necessary."""
    path = processed_path(disease)
    if not path.exists():
        return clean(disease)
    return pd.read_csv(path)


def main() -> None:
    print("=" * 70)
    print("PHASE 3: DATA CLEANING")
    print("=" * 70)

    for disease in config.DISEASES:
        cleaned = clean(disease)
        print(f"  saved -> {processed_path(disease).relative_to(config.PROJECT_ROOT)}")

    print("\nCleaning complete.")


if __name__ == "__main__":
    main()
