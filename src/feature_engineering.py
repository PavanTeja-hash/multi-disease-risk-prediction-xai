"""
Phase 5 - Feature Engineering.

Three jobs, in order:

1. Build derived features that encode medical knowledge the raw columns do not
   express on their own (pulse pressure, BMI category, risk-factor counts).
2. Split into train and test sets *before* any scaling or resampling, so no
   information from the test set can leak into training.
3. Scale features, and apply SMOTE only where the target is genuinely
   imbalanced - and only to the training half.

The order matters more than the individual steps. Scaling or resampling before
splitting is the classic way to leak test data into a model and report an
accuracy that collapses in production.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
import data_cleaning


@dataclass
class Dataset:
    """Everything downstream phases need for one disease."""

    disease: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler
    smote_applied: bool
    # Unscaled test features, kept for explanations: SHAP output is far easier
    # to read against "BMI = 34" than against a standardised "BMI = 1.7".
    X_test_raw: pd.DataFrame
    # Training data as it was *before* scaling and SMOTE. Cross-validation and
    # grid search need this: each fold has to fit its own scaler and run its own
    # resampling, otherwise statistics from the held-out fold leak in through
    # the scaler and SMOTE - the same leakage the train/test split exists to
    # prevent, just hidden one level deeper.
    X_train_raw: pd.DataFrame
    y_train_raw: np.ndarray

    @property
    def label(self) -> str:
        return config.DISEASE_LABEL[self.disease]


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------
def engineer_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the CDC diabetes data."""
    df = df.copy()

    # Poor mental and physical health days are recorded separately but describe
    # the same underlying burden; their sum is a stronger single signal.
    df["unhealthy_days"] = df["MentHlth"] + df["PhysHlth"]

    # Count the positive lifestyle behaviours into one score, so a model can
    # use "generally healthy habits" rather than four scattered flags.
    df["lifestyle_score"] = (
        df["PhysActivity"] + df["Fruits"] + df["Veggies"]
        + (1 - df["Smoker"]) + (1 - df["HvyAlcoholConsump"])
    )

    # Number of major clinical risk factors present.
    df["risk_factor_count"] = (
        df["HighBP"] + df["HighChol"] + (df["BMI"] >= 30).astype(int)
        + df["Smoker"] + df["Stroke"] + df["HeartDiseaseorAttack"]
    )

    # BMI bands as an ordered category (0=under ... 4=obese III).
    df["bmi_category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 35, 100], labels=False
    ).astype(int)

    return df


def engineer_heart(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the cardiovascular data."""
    df = df.copy()

    # Pulse pressure (systolic minus diastolic) is a recognised marker of
    # arterial stiffness and is not recoverable by a linear model from the two
    # readings alone in the way a clinician would use it.
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    # Blood pressure stage, following standard clinical thresholds.
    df["bp_category"] = np.select(
        [
            (df["ap_hi"] < 120) & (df["ap_lo"] < 80),   # normal
            (df["ap_hi"] < 130) & (df["ap_lo"] < 80),   # elevated
            (df["ap_hi"] < 140) | (df["ap_lo"] < 90),   # stage 1
            (df["ap_hi"] < 180) | (df["ap_lo"] < 120),  # stage 2
        ],
        [0, 1, 2, 3],
        default=4,  # hypertensive crisis
    )

    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 35, 100], labels=False
    ).astype(int)

    df["age_group"] = pd.cut(
        df["age_years"], bins=[0, 40, 50, 55, 60, 120], labels=False
    ).astype(int)

    # Both raw lifestyle flags rolled into one score, mirroring the diabetes side.
    df["lifestyle_score"] = (
        df["active"] + (1 - df["smoke"]) + (1 - df["alco"])
    )

    return df


ENGINEERS = {
    config.DIABETES: engineer_diabetes,
    config.HEART: engineer_heart,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def prepare(disease: str, verbose: bool = True) -> Dataset:
    """Run the full feature pipeline for one disease and return a Dataset."""
    target = config.TARGET_COLUMN[disease]

    frame = data_cleaning.load_clean(disease)
    before_columns = frame.shape[1]
    frame = ENGINEERS[disease](frame)

    X = frame.drop(columns=target)
    y = frame[target].to_numpy()
    feature_names = list(X.columns)

    if verbose:
        print(f"\n{config.DISEASE_LABEL[disease]}")
        added = [c for c in feature_names if c not in
                 data_cleaning.load_clean(disease).columns]
        print(f"  engineered      : {before_columns - 1} -> {len(feature_names)} "
              f"features (added {', '.join(added)})")

    # --- Split FIRST, three ways. Everything after this is fitted on train only.
    #
    # A separate validation set exists so the decision threshold can be tuned
    # (Phase 7) without ever looking at the test set. Tuning a threshold on the
    # test set and then reporting test scores is a subtle form of leakage: the
    # reported numbers would be optimistic and would not survive deployment.
    #
    # Critically, the validation set keeps its *natural* class balance. SMOTE is
    # applied only to the training portion. A threshold tuned on 50/50
    # resampled data would be calibrated for a world that does not exist -
    # in reality only 14% of people have diabetes.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # keep the same positive rate in every split
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config.VAL_SIZE_OF_REMAINDER,
        random_state=config.RANDOM_STATE,
        stratify=y_temp,
    )
    if verbose:
        print(f"  split           : {len(X_train):,} train / {len(X_val):,} val / "
              f"{len(X_test):,} test (stratified)")

    X_test_raw = X_test.copy()
    X_train_raw = X_train.copy()
    y_train_raw = y_train.copy()

    # --- Scale. The scaler learns its mean and standard deviation from the
    # training data only, then applies those same numbers to val and test. ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Resample, only if imbalanced, and only the training half. ---
    minority_share = float(y_train.mean())
    minority_share = min(minority_share, 1 - minority_share)
    smote_applied = minority_share < config.IMBALANCE_RATIO_THRESHOLD

    if smote_applied:
        before = len(y_train)
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        # SMOTE appends every synthetic sample to the end, leaving the training
        # array sorted by class. Anything that later takes a contiguous slice -
        # Keras's validation_split takes the *last* 15% before shuffling - would
        # get a single-class validation set and a meaningless AUC. Shuffling
        # here guarantees a mixed slice wherever the data is used.
        shuffle_index = np.random.RandomState(config.RANDOM_STATE).permutation(
            len(y_train)
        )
        X_train_scaled = X_train_scaled[shuffle_index]
        y_train = y_train[shuffle_index]

        if verbose:
            print(f"  SMOTE           : applied (minority was {minority_share:.1%}); "
                  f"train {before:,} -> {len(y_train):,}, now "
                  f"{y_train.mean():.1%} positive")
            print("  shuffled        : SMOTE leaves synthetic rows grouped at the "
                  "end; shuffled so any contiguous slice stays class-balanced")
    elif verbose:
        print(f"  SMOTE           : skipped (minority already {minority_share:.1%}; "
              "resampling would add synthetic rows for no benefit)")

    if verbose:
        print("  note            : split before scaling and resampling; the "
              "validation set keeps its natural balance for threshold tuning")

    return Dataset(
        disease=disease,
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        smote_applied=smote_applied,
        X_test_raw=X_test_raw,
        X_train_raw=X_train_raw,
        y_train_raw=y_train_raw,
    )


def prepare_all(verbose: bool = True) -> dict:
    return {d: prepare(d, verbose=verbose) for d in config.DISEASES}


def main() -> None:
    print("=" * 70)
    print("PHASE 5: FEATURE ENGINEERING")
    print("=" * 70)
    datasets = prepare_all()
    print("\nFeature engineering complete.")
    for disease, ds in datasets.items():
        print(f"  {ds.label:<15} train {ds.X_train.shape}  test {ds.X_test.shape}")


if __name__ == "__main__":
    main()
