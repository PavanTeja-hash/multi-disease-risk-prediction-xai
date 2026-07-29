"""
Hyperparameter tuning with grid search.

Parameters a model *learns* from data are called weights. Parameters you have
to *choose* before training - how many trees, how fast they learn, how much
regularisation - are hyperparameters. Defaults are reasonable guesses by the
library authors, not answers tailored to your data.

Grid search tries every combination on a list you supply, scores each with
cross-validation, and reports the winner. Same pipeline discipline as
cross_validation.py: scaling and SMOTE run inside each fold, so no fold's
statistics leak into the fold it is scored on.

Note on what "best" means here: the grid is scored on ROC-AUC, which measures
how well the model *ranks* patients and is independent of any decision cut-off.
The cut-off itself is a separate decision, tuned in threshold_tuning.py against
a screening target. Ranking first, then deciding.

Usage:
    python src/hyperparameter_tuning.py           # both diseases
    python src/hyperparameter_tuning.py diabetes  # one disease
"""

import sys
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
import cross_validation
import feature_engineering
import train_models

CV_FOLDS = 3  # 3 rather than 5 - grid search multiplies the work by the grid size

# Grids kept deliberately small. A huge grid is slow and mostly explores
# settings that cannot plausibly win.
PARAM_GRIDS = {
    train_models.GRADIENT_BOOSTING: {
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_leaf_nodes": [15, 31, 63],
    },
    train_models.LOGISTIC_REGRESSION: {
        # C controls regularisation strength: small C = simpler model, less
        # able to overfit; large C = more freedom to fit the training data.
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
}


def tune_model(disease: str, model_name: str) -> dict:
    """Grid-search one model on one disease."""
    dataset = feature_engineering.prepare(disease, verbose=False)
    model = train_models.build_models()[model_name]
    pipeline = cross_validation.build_pipeline(
        model, use_smote=dataset.smote_applied
    )
    grid = PARAM_GRIDS[model_name]

    combinations = 1
    for values in grid.values():
        combinations *= len(values)

    print(f"\n  {model_name} on {dataset.label}")
    print(f"    {combinations} combinations x {CV_FOLDS} folds "
          f"= {combinations * CV_FOLDS} fits")

    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=config.RANDOM_STATE),
        n_jobs=1,
        verbose=0,
    )

    start = time.time()
    search.fit(dataset.X_train_raw, dataset.y_train_raw)
    elapsed = time.time() - start

    # Reporting the worst combination alongside the best matters: if they are
    # nearly identical, tuning bought almost nothing and the defaults were
    # already fine. That is a finding, not a failure.
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    print(f"    finished in {elapsed:.0f}s")
    print(f"    best params : {best}")
    print(f"    best ROC-AUC: {search.best_score_:.4f}")
    print(f"    worst in grid: {min(search.cv_results_['mean_test_score']):.4f} "
          f"(spread {search.best_score_ - min(search.cv_results_['mean_test_score']):.4f})")

    return {
        "disease": config.DISEASE_LABEL[disease],
        "model": model_name,
        "best_score": search.best_score_,
        "worst_score": float(min(search.cv_results_["mean_test_score"])),
        "best_params": str(best),
        "seconds": round(elapsed),
    }


def main() -> None:
    diseases = config.DISEASES
    if len(sys.argv) > 1 and sys.argv[1] in config.DISEASES:
        diseases = (sys.argv[1],)

    print("=" * 70)
    print("HYPERPARAMETER TUNING (grid search)")
    print("=" * 70)

    rows = []
    for disease in diseases:
        for model_name in PARAM_GRIDS:
            rows.append(tune_model(disease, model_name))

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    config.ensure_directories()
    path = config.REPORTS_DIR / "hyperparameter_tuning.csv"
    table.to_csv(path, index=False)
    print(f"\n    saved -> reports/{path.name}")


if __name__ == "__main__":
    main()
