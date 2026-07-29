"""
Central configuration for the Multi-Disease Risk Prediction project.

Everything that another module might need to know about *where things live*
or *what the data means* is defined here once, so no other file has to
hardcode a path or repeat a column list.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# config.py lives in src/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw dataset files (as downloaded, never modified)
DIABETES_RAW = DATA_RAW / "cdc_diabetes_health_indicators.csv"
HEART_RAW = DATA_RAW / "cardio_train.csv"

# The heart dataset is semicolon-separated, which is unusual enough to matter.
HEART_SEPARATOR = ";"


def ensure_directories() -> None:
    """Create every output directory the pipeline writes to."""
    for directory in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Disease keys
# ---------------------------------------------------------------------------
# Used throughout the pipeline to loop over both diseases uniformly.
DIABETES = "diabetes"
HEART = "heart"
DISEASES = (DIABETES, HEART)

TARGET_COLUMN = {
    DIABETES: "Diabetes_binary",
    HEART: "cardio",
}

DISEASE_LABEL = {
    DIABETES: "Diabetes",
    HEART: "Heart Disease",
}

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Of the 80% that is not test data, a further 20% becomes the validation set
# used to tune the decision threshold. Final split is 64% / 16% / 20%.
VAL_SIZE_OF_REMAINDER = 0.2

# SMOTE is only applied where the target is genuinely imbalanced. The heart
# dataset is almost exactly 50/50, so resampling it would add synthetic rows
# for no benefit. Deciding this per-dataset (rather than blanket-applying
# SMOTE) is a deliberate choice.
IMBALANCE_RATIO_THRESHOLD = 0.35  # minority share below this triggers SMOTE

# ---------------------------------------------------------------------------
# Human-readable feature names
# ---------------------------------------------------------------------------
# Raw column names are cryptic ("ap_hi", "GenHlth"). The explainability layer
# turns model output into plain English, so it needs real names to work with.
FEATURE_DESCRIPTIONS = {
    # --- CDC diabetes dataset ---
    "HighBP": "high blood pressure",
    "HighChol": "high cholesterol",
    "CholCheck": "cholesterol check in past 5 years",
    "BMI": "body mass index",
    "Smoker": "smoking history",
    "Stroke": "history of stroke",
    "HeartDiseaseorAttack": "history of heart disease or heart attack",
    "PhysActivity": "physical activity",
    "Fruits": "daily fruit consumption",
    "Veggies": "daily vegetable consumption",
    "HvyAlcoholConsump": "heavy alcohol consumption",
    "AnyHealthcare": "has healthcare coverage",
    "NoDocbcCost": "skipped doctor visit due to cost",
    "GenHlth": "self-reported general health (1=excellent, 5=poor)",
    "MentHlth": "days of poor mental health (past 30)",
    "PhysHlth": "days of poor physical health (past 30)",
    "DiffWalk": "difficulty walking or climbing stairs",
    "Sex": "sex",
    "Age": "age group",
    "Education": "education level",
    "Income": "income level",
    # --- Cardiovascular dataset ---
    "age": "age",
    "age_years": "age in years",
    "gender": "gender",
    "height": "height (cm)",
    "weight": "weight (kg)",
    "ap_hi": "systolic blood pressure",
    "ap_lo": "diastolic blood pressure",
    "cholesterol": "cholesterol level (1=normal, 3=well above normal)",
    "gluc": "glucose level (1=normal, 3=well above normal)",
    "smoke": "smoking",
    "alco": "alcohol intake",
    "active": "physically active",
    # --- Engineered features ---
    "bmi": "body mass index",
    "bmi_category": "BMI category",
    "pulse_pressure": "pulse pressure (systolic minus diastolic)",
    "bp_category": "blood pressure category",
    "age_group": "age group",
    "unhealthy_days": "total unhealthy days (past 30)",
    "lifestyle_score": "healthy lifestyle score",
    "risk_factor_count": "number of major risk factors",
}


def describe_feature(name: str) -> str:
    """Return a human-readable description for a feature name."""
    return FEATURE_DESCRIPTIONS.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Cross-disease concept mapping
# ---------------------------------------------------------------------------
# The two datasets measure several of the same clinical concepts under
# different column names, and sometimes in different units - the survey records
# "HighBP" as a yes/no flag while the cardiovascular data records an actual
# systolic reading in "ap_hi". Comparing the two models' feature importance
# directly would find no overlap at all, which is a naming artefact rather than
# a real finding. This maps both vocabularies onto shared concepts so the
# "which risk factors are shared?" question can actually be answered.
CANONICAL_CONCEPT = {
    # blood pressure
    "HighBP": "blood pressure",
    "ap_hi": "blood pressure",
    "ap_lo": "blood pressure",
    "bp_category": "blood pressure",
    "pulse_pressure": "blood pressure",
    # cholesterol
    "HighChol": "cholesterol",
    "cholesterol": "cholesterol",
    "CholCheck": "cholesterol",
    # body mass
    "BMI": "body mass",
    "bmi": "body mass",
    "bmi_category": "body mass",
    "weight": "body mass",
    "height": "body mass",
    # age
    "Age": "age",
    "age_years": "age",
    "age_group": "age",
    # blood sugar
    "gluc": "blood glucose",
    # lifestyle
    "Smoker": "smoking",
    "smoke": "smoking",
    "PhysActivity": "physical activity",
    "active": "physical activity",
    "HvyAlcoholConsump": "alcohol intake",
    "alco": "alcohol intake",
    "lifestyle_score": "overall lifestyle",
    # demographics
    "Sex": "sex",
    "gender": "sex",
}


def canonical_concept(name: str) -> str:
    """Map a dataset-specific column onto a shared clinical concept."""
    return CANONICAL_CONCEPT.get(name, describe_feature(name))


# ---------------------------------------------------------------------------
# Clinically valid ranges (used by the cleaning phase)
# ---------------------------------------------------------------------------
# The cardiovascular dataset contains physically impossible values, e.g.
# systolic blood pressure recorded as -150 or 16020, and heights of 55cm in an
# adult cohort. These are data-entry errors, not rare-but-real patients, so
# they are removed rather than kept as "outliers".
VALID_RANGES = {
    "ap_hi": (70, 250),      # systolic blood pressure, mmHg
    "ap_lo": (40, 200),      # diastolic blood pressure, mmHg
    "height": (120, 220),    # cm, adult cohort
    "weight": (30, 250),     # kg
    "bmi": (12, 70),         # kg/m^2
}
