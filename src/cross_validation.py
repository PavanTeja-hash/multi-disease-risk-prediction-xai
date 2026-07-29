"""
Cross-validation - how stable are these scores, really?

A single train/test split gives one number. Cross-validation gives several, so
you can see whether that number is solid or whether it happened to land well.

The data is cut into 5 folds. The model trains on 4 and is scored on the 5th,
five times over, each fold taking a turn as the held-out set. Report the mean
and the spread.

THE PART THAT IS EASY TO GET WRONG
----------------------------------
Scaling and SMOTE must happen *inside* each fold, fitted only on that fold's
training portion.

The tempting shortcut is to scale and resample the whole training set once, and
then cross-validate. That leaks: the scaler's mean would be computed using rows
that later act as the held-out fold, and SMOTE would generate synthetic patients
from rows that later act as the held-out fold. Scores come out flattering and
mean nothing.

An imbalanced-learn Pipeline solves this. It re-fits every step for each fold,
and - crucially - it applies SMOTE only when *fitting*, never when scoring. So
held-out folds keep their real, natural class balance.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

import config
import feature_engineering
import train_models

N_FOLDS = 5
SCORING = ("roc_auc", "f1", "recall", "precision")


def build_pipeline(model, use_smote: bool) -> ImbPipeline:
    """Wrap a model so scaling and resampling happen per fold, not globally."""
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def cross_validate_disease(disease: str, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Run k-fold cross-validation for every classical model on one disease."""
    dataset = feature_engineering.prepare(disease, verbose=False)

    # Raw training data: unscaled, un-resampled. The pipeline handles both.
    X = dataset.X_train_raw
    y = dataset.y_train_raw

    print(f"\n{dataset.label}")
    print(f"  {len(X):,} training rows, {y.mean():.1%} positive, "
          f"{n_folds}-fold stratified CV")
    print(f"  SMOTE inside each fold: {dataset.smote_applied}")

    folds = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE
    )

    rows = {}
    for name, model in train_models.build_models().items():
        pipeline = build_pipeline(model, use_smote=dataset.smote_applied)
        results = cross_validate(
            pipeline, X, y, cv=folds, scoring=list(SCORING), n_jobs=1,
        )

        row = {}
        for metric in SCORING:
            scores = results[f"test_{metric}"]
            row[metric] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        rows[name] = row

        print(f"    {name:<20} ROC-AUC {row['roc_auc']:.4f} "
              f"(+/- {row['roc_auc_std']:.4f})   "
              f"recall {row['recall']:.4f} (+/- {row['recall_std']:.4f})")

    return pd.DataFrame(rows).T


def main() -> None:
    print("=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)

    config.ensure_directories()
    for disease in config.DISEASES:
        table = cross_validate_disease(disease)
        path = config.REPORTS_DIR / f"{disease}_cross_validation.csv"
        table.round(4).to_csv(path)
        print(f"    saved -> reports/{path.name}")

    print("\nA small standard deviation means the score is stable across folds,")
    print("not an artefact of one lucky split.")


if __name__ == "__main__":
    main()
