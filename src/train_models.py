"""
Phase 6 - Classical machine learning models.

Three models are trained per disease, each chosen for a reason:

* Logistic Regression - the baseline. Simple, fast, and fully interpretable on
  its own. If a complex model cannot beat it, the complexity is not justified.
* Linear SVM - finds the widest possible margin between the two classes.
* Gradient Boosting - builds trees sequentially, each correcting the previous
  one's mistakes. Usually the strongest of the three on tabular data.

Note on the SVM: a standard RBF-kernel SVM scales roughly quadratically with
the number of samples, which is not viable on a 348k-row training set. A
linear-kernel SVM solves the same maximum-margin problem in a form that scales
to this size. It is wrapped in CalibratedClassifierCV because the raw SVM
outputs distances from the decision boundary, not probabilities, and every
downstream step (ROC-AUC, SHAP, the risk report) needs a probability.
"""

import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import config
import feature_engineering

LOGISTIC_REGRESSION = "Logistic Regression"
SVM = "Linear SVM"
GRADIENT_BOOSTING = "Gradient Boosting"

MODEL_ORDER = (LOGISTIC_REGRESSION, SVM, GRADIENT_BOOSTING)


def build_models() -> dict:
    """Create one fresh, untrained instance of each model."""
    return {
        LOGISTIC_REGRESSION: LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        ),
        SVM: CalibratedClassifierCV(
            LinearSVC(
                # dual=False is the right solver when rows far outnumber
                # columns, which is the case for both datasets.
                dual=False,
                C=1.0,
                random_state=config.RANDOM_STATE,
            ),
            cv=3,
        ),
        GRADIENT_BOOSTING: HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=None,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=config.RANDOM_STATE,
        ),
    }


def model_path(disease: str, model_name: str):
    slug = model_name.lower().replace(" ", "_")
    return config.MODELS_DIR / f"{disease}_{slug}.joblib"


def train(dataset: feature_engineering.Dataset, save: bool = True) -> dict:
    """Fit every model on one disease's training data."""
    print(f"\n{dataset.label}  ({dataset.X_train.shape[0]:,} training rows)")
    config.ensure_directories()

    trained = {}
    for name, model in build_models().items():
        start = time.time()
        model.fit(dataset.X_train, dataset.y_train)
        elapsed = time.time() - start

        # A quick training-set accuracy is printed only as a sanity check that
        # fitting actually happened. Real evaluation is Phase 7, on the test set.
        train_accuracy = model.score(dataset.X_train, dataset.y_train)
        print(f"  {name:<20} fitted in {elapsed:6.1f}s   "
              f"(train accuracy {train_accuracy:.3f})")

        trained[name] = model
        if save:
            joblib.dump(model, model_path(dataset.disease, name))

    return trained


def load(disease: str, model_name: str):
    path = model_path(disease, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found - run src/train_models.py first."
        )
    return joblib.load(path)


def load_all(disease: str) -> dict:
    """Load every saved classical model for one disease."""
    return {name: load(disease, name) for name in MODEL_ORDER}


def main() -> None:
    print("=" * 70)
    print("PHASE 6: MACHINE LEARNING MODELS")
    print("=" * 70)

    datasets = feature_engineering.prepare_all(verbose=False)
    for disease in config.DISEASES:
        train(datasets[disease])

    print(f"\nModels saved to {config.MODELS_DIR.name}/")


if __name__ == "__main__":
    main()
