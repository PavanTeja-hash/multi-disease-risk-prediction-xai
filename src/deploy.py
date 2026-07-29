"""
Package the trained pipeline into a single loadable artefact per disease.

The demo app must not re-run feature engineering or re-read the raw data - it
needs exactly what is required to score one patient: the model, the scaler
fitted on the training data, the feature order, the tuned decision threshold,
and sensible defaults for any field the user does not fill in.

Bundling those together avoids the classic deployment bug where the app builds
its input vector in a different column order than the model was trained on, and
silently returns nonsense.
"""

import joblib
import numpy as np
import pandas as pd

import config
import data_cleaning
import feature_engineering
import threshold_tuning
import train_models

BUNDLE_VERSION = 2  # v2 stores raw-column defaults so serving can re-engineer


def bundle_path(disease: str):
    return config.MODELS_DIR / f"{disease}_bundle.joblib"


def build(disease: str, target_recall: float = 0.80) -> dict:
    """Assemble everything needed to score a single patient."""
    dataset = feature_engineering.prepare(disease, verbose=False)
    model = train_models.load(disease, train_models.GRADIENT_BOOSTING)

    # Tune the threshold on validation data only - never on the test set.
    val_probabilities = model.predict_proba(dataset.X_val)[:, 1]
    threshold = threshold_tuning.threshold_for_recall(
        dataset.y_val, val_probabilities, target_recall
    )

    # Two sets of defaults, for two different jobs.
    #
    # `defaults` covers every column the model consumes, engineered ones
    # included, and exists only so older callers keep working.
    #
    # `raw_defaults` covers the columns as they appear *before* feature
    # engineering. This is the one that matters: at serving time the app
    # supplies raw measurements, so the engineered columns must be recomputed
    # from them rather than filled in with medians. Filling them with medians
    # is how training/serving skew happens - the model would receive BMI=45
    # alongside a bmi_category of "overweight", two facts that contradict each
    # other, and quietly return a distorted score.
    raw_columns = [
        column for column in data_cleaning.load_clean(disease).columns
        if column != config.TARGET_COLUMN[disease]
    ]
    raw_defaults = (
        data_cleaning.load_clean(disease)[raw_columns].median().to_dict()
    )

    bundle = {
        "version": BUNDLE_VERSION,
        "disease": disease,
        "label": config.DISEASE_LABEL[disease],
        "model": model,
        "scaler": dataset.scaler,
        "feature_names": dataset.feature_names,
        "raw_feature_names": raw_columns,
        "threshold": float(threshold),
        "target_recall": target_recall,
        "defaults": dataset.X_test_raw.median().to_dict(),
        "raw_defaults": raw_defaults,
        "background": dataset.X_train[:200],  # for SHAP, if needed
    }

    config.ensure_directories()
    joblib.dump(bundle, bundle_path(disease))
    size_mb = bundle_path(disease).stat().st_size / 1_000_000
    print(f"  {config.DISEASE_LABEL[disease]:<15} threshold {threshold:.3f}  "
          f"-> models/{bundle_path(disease).name}  ({size_mb:.1f} MB)")
    return bundle


def load(disease: str) -> dict:
    path = bundle_path(disease)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found - run `python src/deploy.py` first."
        )
    return joblib.load(path)


def score(bundle: dict, patient: dict) -> dict:
    """
    Score one patient given a dict of raw (unscaled) measurements.

    The order here is the whole point:

        1. start from the raw columns, filling anything the caller omitted
           with the dataset median
        2. run the SAME feature-engineering function used in training
        3. select the model's features, in the model's own order

    Step 2 is what prevents training/serving skew. Derived columns like
    bmi_category, pulse_pressure and risk_factor_count are recomputed from the
    caller's actual measurements, so they can never disagree with the raw
    values they are supposed to summarise.
    """
    if "raw_defaults" in bundle:
        raw_row = {
            name: patient.get(name, bundle["raw_defaults"][name])
            for name in bundle["raw_feature_names"]
        }
        # Any extra keys the caller supplied (already-engineered values, say)
        # are honoured rather than silently dropped.
        raw_row.update({k: v for k, v in patient.items() if k in raw_row})

        engineered = feature_engineering.ENGINEERS[bundle["disease"]](
            pd.DataFrame([raw_row])
        )
        # Anything still missing (rare) falls back to the full default set.
        for name in bundle["feature_names"]:
            if name not in engineered.columns:
                engineered[name] = bundle["defaults"][name]
        frame = engineered[bundle["feature_names"]]
    else:
        # Bundle predates version 2 - old behaviour, kept so stale artefacts
        # do not crash. Regenerate with `python src/deploy.py`.
        row = {
            name: patient.get(name, bundle["defaults"][name])
            for name in bundle["feature_names"]
        }
        frame = pd.DataFrame([row], columns=bundle["feature_names"])

    scaled = bundle["scaler"].transform(frame)
    probability = float(bundle["model"].predict_proba(scaled)[0, 1])

    return {
        "disease": bundle["disease"],
        "label": bundle["label"],
        "probability": probability,
        "flagged": bool(probability >= bundle["threshold"]),
        "threshold": bundle["threshold"],
        "raw_row": frame.iloc[0],
        "scaled_row": scaled,
    }


def explain(bundle: dict, scored: dict, top_n: int = 6) -> dict:
    """SHAP explanation for a scored patient, in report_generator's format."""
    import shap

    import explainability

    explainer = shap.TreeExplainer(bundle["model"])
    contributions = explainability._positive_class_values(
        explainer.shap_values(scored["scaled_row"])
    )[0]

    frame = pd.DataFrame({
        "feature": bundle["feature_names"],
        "value": [scored["raw_row"][f] for f in bundle["feature_names"]],
        "shap": contributions,
    })
    frame["magnitude"] = frame["shap"].abs()
    frame = frame.sort_values("magnitude", ascending=False)
    top = frame.head(top_n)

    return {
        "disease": bundle["disease"],
        "probability": scored["probability"],
        # The report's wording has to agree with the flag the app shows, so the
        # decision threshold travels with the explanation.
        "threshold": bundle["threshold"],
        "actual": None,
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


def main() -> None:
    print("=" * 70)
    print("PACKAGING DEPLOYABLE BUNDLES")
    print("=" * 70)
    for disease in config.DISEASES:
        build(disease)
    print("\nBundles ready. Launch the demo with:  streamlit run app.py")


if __name__ == "__main__":
    main()
