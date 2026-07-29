"""
Interactive demo: enter a patient's details, get both risk scores explained.

    streamlit run app.py

The point of the demo is the explanation, not the number. Anyone can show a
percentage; this shows which measurements produced it and says so in English.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st

import config  # noqa: E402
import deploy  # noqa: E402
import report_generator  # noqa: E402

# Streamlit Cloud supplies secrets through st.secrets, but report_generator
# reads API keys from the environment so it also works from a plain terminal.
# Copy them across so one code path serves both. Accessing st.secrets raises
# when no secrets file exists (the normal local case), hence the guard.
try:
    for _key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass  # no secrets configured - the template narrator still works

st.set_page_config(page_title="MediRisk", page_icon="+", layout="wide")


@st.cache_resource
def load_bundles():
    return {d: deploy.load(d) for d in config.DISEASES}


def risk_colour(probability: float) -> str:
    if probability >= 0.70:
        return "#C1444F"
    if probability >= 0.40:
        return "#E08A2E"
    return "#4C9F70"


def contribution_chart(explanation: dict, title: str):
    frame = explanation["all_contributions"].head(7).iloc[::-1]
    labels = [
        f"{config.describe_feature(f).split(' (')[0]} = {v:g}"
        for f, v in zip(frame["feature"], frame["value"])
    ]
    colours = ["#C1444F" if v > 0 else "#4C9F70" for v in frame["shap"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(labels, frame["shap"], color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effect on risk  (right = raises)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
st.title("MediRisk")
st.markdown("**Explainable AI for multi-disease risk prediction and screening**")
st.caption(
    "Screens for diabetes and heart disease from one set of measurements, and "
    "explains every prediction. The two conditions are clinically linked - in "
    "the CDC survey data used here, diabetics have heart disease at 3.0x the "
    "rate of non-diabetics."
)

try:
    bundles = load_bundles()
except FileNotFoundError as error:
    st.error(f"{error}\n\nRun `python main.py` first to train and package the models.")
    st.stop()

with st.sidebar:
    st.header("Patient details")

    st.subheader("Basics")
    age = st.slider("Age", 30, 80, 55)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    height = st.slider("Height (cm)", 140, 210, 170)
    weight = st.slider("Weight (kg)", 40, 180, 85)
    bmi = round(weight / (height / 100) ** 2, 1)
    st.caption(f"BMI: **{bmi}**")

    st.subheader("Measurements")
    systolic = st.slider("Systolic blood pressure", 90, 220, 140)
    diastolic = st.slider("Diastolic blood pressure", 50, 140, 90)
    cholesterol = st.select_slider(
        "Cholesterol", [1, 2, 3],
        format_func=lambda v: {1: "Normal", 2: "Above normal",
                               3: "Well above normal"}[v], value=1,
    )
    glucose = st.select_slider(
        "Blood glucose", [1, 2, 3],
        format_func=lambda v: {1: "Normal", 2: "Above normal",
                               3: "Well above normal"}[v], value=1,
    )

    st.subheader("History and lifestyle")
    general_health = st.select_slider(
        "General health", [1, 2, 3, 4, 5],
        format_func=lambda v: {1: "Excellent", 2: "Very good", 3: "Good",
                               4: "Fair", 5: "Poor"}[v], value=3,
    )
    smoker = st.checkbox("Smoker")
    active = st.checkbox("Physically active", value=True)
    difficulty_walking = st.checkbox("Difficulty walking or climbing stairs")
    prior_stroke = st.checkbox("History of stroke")
    heavy_alcohol = st.checkbox("Heavy alcohol consumption")

    submitted = st.button("Assess risk", type="primary", use_container_width=True)

if not submitted:
    st.info("Fill in the patient details on the left, then select **Assess risk**.")
    st.stop()

# --- Map the form onto each dataset's own vocabulary --------------------------
high_bp = int(systolic >= 140 or diastolic >= 90)
high_chol = int(cholesterol >= 2)

# The CDC survey bins age into 13 groups; the cardiovascular dataset stores age
# in years directly, so the form's single "Age" field has to be translated.
#
# The first bucket is 7 years wide (18-24) and every later one is 5, so a plain
# (age - 18) // 5 drifts by one bucket from age 33 onward - it aged a
# 33-year-old into the 35-39 band and inflated their diabetes risk. Anchoring
# the arithmetic at 25, where the regular 5-year steps actually begin, avoids it.
age_group = 1 if age <= 24 else min(13, (age - 25) // 5 + 2)

patients = {
    config.DIABETES: {
        "BMI": bmi, "GenHlth": general_health, "HighBP": high_bp,
        "HighChol": high_chol, "Age": age_group, "Sex": int(sex == "Male"),
        "Smoker": int(smoker), "PhysActivity": int(active),
        "DiffWalk": int(difficulty_walking), "Stroke": int(prior_stroke),
        "HvyAlcoholConsump": int(heavy_alcohol),
    },
    config.HEART: {
        "age_years": float(age), "gender": 2 if sex == "Male" else 1,
        "height": float(height), "weight": float(weight), "bmi": bmi,
        "ap_hi": float(systolic), "ap_lo": float(diastolic),
        "cholesterol": cholesterol, "gluc": glucose,
        "smoke": int(smoker), "active": int(active),
        "alco": int(heavy_alcohol),
    },
}

# --- Score, explain, report ---------------------------------------------------
narrator = report_generator.get_narrator()
columns = st.columns(2)

for column, disease in zip(columns, config.DISEASES):
    bundle = bundles[disease]
    scored = deploy.score(bundle, patients[disease])
    explanation = deploy.explain(bundle, scored)

    with column:
        st.markdown(f"### {bundle['label']}")
        probability = scored["probability"]
        st.markdown(
            f"<div style='font-size:2.8rem;font-weight:700;"
            f"color:{risk_colour(probability)}'>{probability:.0%}</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(probability, 1.0))

        if scored["flagged"]:
            st.warning(
                f"Flagged for follow-up (screening threshold "
                f"{scored['threshold']:.2f}, tuned for "
                f"{bundle['target_recall']:.0%} recall)"
            )
        else:
            st.success(
                f"Below the screening threshold ({scored['threshold']:.2f})"
            )

        st.pyplot(contribution_chart(explanation, "What drove this score"))

        with st.expander("Plain-English report", expanded=True):
            st.text(report_generator.generate_report(
                explanation, narrator=narrator))

st.divider()
st.caption(
    "Screening estimates from models trained on the CDC Diabetes Health "
    "Indicators dataset (253,680 people) and the Cardiovascular Disease "
    "dataset (70,000 people). These are population-level statistical "
    "estimates, not a diagnosis, and the two models are trained on different "
    "populations so their scores are not directly comparable to each other."
)
