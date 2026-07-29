# Multi-Disease Risk Prediction with Explainable AI

### ▶ [Try the live app](https://multi-disease-risk-prediction-xai-tdg7zjqb3huyahcf3eqk2q.streamlit.app/)

Enter a patient's measurements and get both risk scores, the factors that drove
each one, and a plain-English explanation. *(Free hosting — the app sleeps when
idle, so the first load can take ~30 seconds.)*

Predicts a patient's risk of **diabetes** and **heart disease** from one set of
measurements, and explains every prediction in plain English instead of
returning a bare percentage.

Most disease-prediction projects model one condition and stop at an accuracy
score. This one covers two clinically linked conditions in a single pipeline,
tunes its decision threshold for screening rather than for accuracy, and uses
SHAP and LIME to say *why* each patient scored the way they did.

---

## The premise, tested on real data

The project rests on the claim that diabetes and heart disease are linked. The
CDC survey records both conditions for the same 253,680 respondents, so the
claim is measured rather than cited:

| Group | Heart disease prevalence |
|---|---|
| Non-diabetic | 7.3% |
| Diabetic | **22.3%** |

**Diabetics have heart disease at 3.0x the rate of non-diabetics** — matching
the 2–4x range reported in the clinical literature, and justifying treating the
two as one linked problem.

---

## Results

Both models are evaluated on a held-out test set that was never used for
training or for threshold tuning.

### Diabetes (50,620 test patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.734 | 0.313 | 0.761 | 0.444 | 0.822 |
| Linear SVM | 0.734 | 0.313 | 0.761 | 0.444 | 0.822 |
| **Gradient Boosting** | 0.865 | 0.535 | 0.211 | 0.303 | **0.825** |
| Neural Network | 0.758 | 0.327 | 0.699 | 0.446 | 0.815 |

### Heart disease (13,743 test patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.726 | 0.756 | 0.658 | 0.704 | 0.792 |
| Linear SVM | 0.726 | 0.754 | 0.662 | 0.705 | 0.791 |
| **Gradient Boosting** | 0.734 | 0.760 | 0.676 | 0.716 | **0.803** |
| Neural Network | 0.733 | 0.753 | 0.685 | 0.717 | 0.802 |

### The accuracy trap, and the fix

The diabetes table above contains the single most important result in the
project. Gradient Boosting has the **best ROC-AUC and the best accuracy** — and
at the default 0.5 cut-off it catches only **21% of actual diabetics**. It
scores 86.5% accuracy largely by predicting "healthy" for almost everyone,
because 86% of the population *is* healthy.

Accuracy was the wrong target. Ranking was never the problem; the cut-off was.
Re-tuning the threshold on a held-out validation set (never the test set) fixes
it:

| Disease | Threshold | Recall | Precision | Real patients found |
|---|---|---|---|---|
| Diabetes | 0.50 → **0.148** | 0.211 → **0.805** | 0.535 → 0.297 | 1,488 → **5,677** of 7,054 |
| Heart disease | 0.50 → **0.389** | 0.676 → **0.793** | 0.760 → 0.690 | 4,596 → **5,390** of 6,801 |

Nearly **4x more diabetics found**, at the cost of more false alarms — the
correct trade for a screening tool, where a missed case is far more expensive
than an unnecessary follow-up. The threshold was chosen to hit 80% recall on
validation data and achieved 80.5% and 79.3% on the test set, which is evidence
the tuning generalised rather than leaked.

### What screening costs

Recall is a clinical decision with an operational price, made explicit:

| Disease | Recall target | Achieved | Precision | Flagged per 1,000 | Cases missed |
|---|---|---|---|---|---|
| Diabetes | 70% | 0.710 | 0.338 | 293 | 2,043 |
| Diabetes | 80% | 0.805 | 0.297 | 378 | 1,377 |
| Diabetes | 90% | 0.903 | 0.250 | 505 | 681 |
| Heart disease | 70% | 0.692 | 0.752 | 456 | 2,093 |
| Heart disease | 80% | 0.793 | 0.690 | 568 | 1,411 |
| Heart disease | 90% | 0.895 | 0.617 | 718 | 713 |

---

## Shared risk factors

Because both diseases run through the same pipeline, their SHAP importances can
be mapped onto common clinical concepts and compared directly — something a
single-disease project cannot produce:

| Risk factor | Diabetes | Heart disease |
|---|---|---|
| Blood pressure | 5.9% | **56.1%** |
| Age | 15.4% | 16.2% |
| Body mass | **14.5%** | 7.7% |
| Cholesterol | 5.9% | 12.1% |

Blood pressure dominates heart disease; body mass matters more for diabetes;
age matters equally for both. The strongest driver unique to one disease is
**income level (14.2%)** for diabetes — a socioeconomic signal with no
equivalent on the cardiovascular side.

Highest-risk groups found in the data:

- **Diabetes** — poor self-reported health + BMI 35+: **56.4% diabetic** (2,657 people)
- **Heart disease** — stage 2 hypertension, any age: **85–88% affected**

---

## How it works

```
data/raw/           two public datasets, never modified
  ↓ data_cleaning   remove impossible values, repair swapped readings
  ↓ eda             measure the disease link, find the patterns
  ↓ feature_eng     derive features, split 64/16/20, scale, SMOTE
  ↓ train_models    Logistic Regression · Linear SVM · Gradient Boosting
  ↓ train_nn        feedforward neural network (TensorFlow/Keras)
  ↓ cross_val       5-fold CV, resampling inside each fold
  ↓ tuning          grid search over the main hyperparameters
  ↓ evaluate        metrics, ROC curves, confusion matrices
  ↓ threshold       tune the cut-off on validation data
  ↓ explainability  SHAP (global + per-patient) · LIME cross-check
  ↓ report_gen      plain-English report per patient
  ↓ insights        risk segments, shared drivers, screening cost
  ↓ deploy          package model + scaler + threshold into one artefact
```

### Decisions worth explaining in an interview

**Repeated survey rows were kept, not dropped.** 24,206 rows in the diabetes
data are exact duplicates. `drop_duplicates()` is the reflex, and it is wrong
here: duplicated rows are only **1.35%** diabetic versus **15.99%** for unique
rows, because they are overwhelmingly healthy respondents who happen to give
identical answers to a 21-question survey. Dropping them would delete mostly
negative cases and inflate the apparent disease rate.

**SMOTE was applied to one dataset, not both.** Diabetes is 13.9% positive, so
it was resampled. Heart disease is 49.5% positive — already balanced, so
resampling it would add synthetic patients for no benefit. Applying a technique
because it is on the syllabus, rather than because the data needs it, is how
pipelines get worse.

**A three-way split, because threshold tuning needs its own data.** Tuning the
cut-off on the test set and then reporting test scores is a subtle leak: the
number would be fitted to the data it claims to measure. The validation set
also keeps its natural class balance — a threshold tuned on SMOTE-balanced
50/50 data would be calibrated for a population that does not exist.

**A linear-kernel SVM, deliberately.** An RBF-kernel SVM scales roughly
quadratically and is not viable on 279k training rows. The linear kernel solves
the same maximum-margin problem at a workable cost, wrapped in
`CalibratedClassifierCV` because SVMs output distances, not probabilities, and
every downstream step needs a probability.

**The neural network did not win, and that is reported.** It ties gradient
boosting on heart disease and loses on diabetes. Neural networks frequently do
not beat gradient boosting on tabular data; knowing when the more complex model
is not the answer matters more than always reaching for it.

### How stable are these numbers?

5-fold cross-validation, with scaling and SMOTE applied **inside each fold** via
an imbalanced-learn pipeline — resampling the whole training set first would
leak the held-out fold's statistics through the scaler.

| | Diabetes ROC-AUC | Heart ROC-AUC |
|---|---|---|
| Logistic Regression | 0.8252 ± 0.0030 | 0.7888 ± 0.0073 |
| Linear SVM | 0.8252 ± 0.0030 | 0.7883 ± 0.0072 |
| **Gradient Boosting** | **0.8291** ± 0.0028 | **0.7980** ± 0.0075 |

The spread is ±0.003, so these results are not an artefact of one lucky split.
Gradient Boosting's cross-validated recall (0.2087) also reproduces its
single-split recall (0.211), confirming the 21%-recall problem was real rather
than noise.

Grid search over learning rate and tree size moved ROC-AUC by at most **0.005**,
and by 0.0001 for logistic regression — so the defaults were already
near-optimal and were kept. Combined with all four model families landing within
0.01 AUC of each other, that points at the real ceiling: **the features, not the
model.** Meaningful gains would need clinical measurements like HbA1c, not a
better algorithm.

### Three bugs worth knowing about

None of these crashed. Each produced a plausible-looking number that happened to
be wrong, and each was caught by checking a result that didn't smell right.

1. **The neural network validated against a single class.** Keras's
   `validation_split` takes the *last* 15% of the array before shuffling, and
   SMOTE appends every synthetic sample to the end — so the validation set was
   100% positive and AUC was a meaningless 0.0000. Shuffling after resampling
   took validation AUC from 0.0000 to **0.8667**.

2. **LIME explained nothing.** Its background sample came from the
   SMOTE-balanced training set, so it perturbed within a population that is 50%
   synthetic diabetics, the model saturated, and the local fit scored
   R² = 0.005. Switching the background to the natural-distribution validation
   set and selecting the neighbourhood width by local fidelity took R² to
   **0.961**.

3. **Training/serving skew in the deployed app.** The form collected 11 raw
   measurements, but the model expects 25 features — so the 14 engineered ones
   (`bmi_category`, `pulse_pressure`, `risk_factor_count`, …) were being filled
   with dataset medians. The model received `BMI=45` alongside a `bmi_category`
   of "overweight": two facts contradicting each other. Diabetes risk came out
   non-monotonic in BMI, and a severely unwell profile scored *lower* than a
   borderline one. Serving now runs the same feature-engineering function as
   training, so derived columns can never disagree with the measurements they
   summarise.

   Every offline metric was valid throughout — training never had the problem.
   It was visible only by running the deployed app and reading the output.

### On SHAP values

They are reported in **log-odds, not percentage points**. A contribution of
+0.8 means that feature pushed the log-odds up by 0.8 — not "+80% risk". The
model's baseline is also high (~82%) because it was trained on SMOTE-balanced
data, so most real patients get pushed *down* from it. Factors that lower the
score are therefore described as lowering it "relative to the model's average
patient", never as protective — calling a low income protective would be an
unsupported medical claim.

---

## Running it

```bash
python -m venv venv
venv\Scripts\activate                # Windows;  source venv/bin/activate on Unix
pip install -r requirements-dev.txt  # full pipeline, including TensorFlow

python scripts/download_data.py      # fetch both datasets (~25 MB)
python main.py                       # run every phase end to end
streamlit run app.py                 # interactive demo
```

`python main.py --quick` skips the slow phases. Every phase also runs on its
own — `python src/eda.py`, `python src/evaluate.py`, and so on.

**Two requirements files.** `requirements.txt` holds only what the deployed app
needs; `requirements-dev.txt` adds TensorFlow, seaborn, LIME and ucimlrepo for
the full pipeline. The split exists because the demo serves the Gradient
Boosting bundles and never imports the neural network, so shipping TensorFlow to
production is dead weight — and Streamlit Cloud runs Python 3.14, for which
TensorFlow publishes no wheels at all.

### Optional: LLM-written reports

Reports come from templates by default — no API key, no network access, works
for anyone who clones the repo. Setting an API key switches to LLM-written prose
generated from the same SHAP facts:

```bash
set GEMINI_API_KEY=...        # Windows;  export on Unix. Free tier at aistudio.google.com
```

Google Gemini is the default (its free tier needs no billing setup); Anthropic
is used instead if `ANTHROPIC_API_KEY` is set. The model defaults to
`gemini-flash-lite-latest` — the non-lite variant runs an internal reasoning
pass that burned 513 thinking tokens to produce 57 tokens of output, which
wastes free-tier quota and can return an empty response if the budget is spent
before any text is written.

The language model never sees the classifier and is never asked to assess risk.
It receives already-computed SHAP facts and rephrases them, under a system
prompt that forbids inventing factors, giving medical advice, or describing
correlations as causes. Any failure — missing key, rate limit, network error —
falls back to the template.

---

## Data

| Dataset | Rows | Source |
|---|---|---|
| CDC Diabetes Health Indicators | 253,680 | UCI ML Repository (id 891), from the CDC BRFSS survey |
| Cardiovascular Disease | 70,000 | Kaggle (sulianova), fetched from a checksum-verified mirror |

Both cover broad general populations. After cleaning: 253,096 and 68,713 rows.

---

## Limitations

- The two models are trained on **different populations**, so their scores are
  not directly comparable to one another. A patient scoring 60% for both is not
  equally at risk of both.
- The cardiovascular dataset's `cardio` label means "cardiovascular disease
  present" without specifying which condition.
- The diabetes data is **self-reported survey data** — respondents may not know
  they are diabetic, so the true rate is likely higher than 13.9%.
- **The link between the two diseases is evidence, not a mechanism.** It
  justifies screening for both from one set of measurements, and heart disease
  history is a feature in the diabetes model — but the two predictors are
  independent. Modelling the interaction directly would need a single cohort
  labelled for both conditions, which no public dataset provides.
- Both are **screening tools**, not diagnostics. Every output is a
  population-level statistical estimate and needs a clinician in the loop.

### One learned association that should not be read as advice

Heavy alcohol consumption is associated with **lower** diabetes risk in this
data — 5.9% of heavy drinkers are diabetic versus 14.4% of everyone else — and
the model reflects that. It is not protective. Three things produce it:

- **Reverse causation.** Diagnosed diabetics are advised to cut down, so the
  survey captures diabetics who have already stopped. The arrow runs
  *diabetes → less drinking*, and cross-sectional data cannot tell the
  difference.
- **Age confounding.** Heavy drinking falls from 7.8% to 2.8% across the age
  brackets while diabetes climbs from 1.4% to 18.5%. Age drives both.
- **Sparse support.** Only **34** training examples resemble a high-risk heavy
  drinker, and those 34 show half the effect the model extrapolates.

This is exactly the failure mode explainable AI exists to surface, and it is why
every generated report says "correlated with" rather than "caused by".

---

## Layout

```
data/raw/          original datasets (not committed)
data/processed/    cleaned data
src/               pipeline, one module per phase
requirements.txt   runtime deps for the deployed app
requirements-dev.txt  full pipeline deps (adds TensorFlow, seaborn, LIME)
models/            trained models and deployable bundles
reports/figures/   26 generated figures
reports/           metric tables, screening workload
scripts/           dataset download
main.py            runs everything
app.py             Streamlit demo
```

**Stack:** Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Google Gemini,
imbalanced-learn, SHAP, LIME, Matplotlib, Seaborn, Streamlit.
