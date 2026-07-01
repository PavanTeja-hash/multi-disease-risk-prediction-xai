# Multi-Disease Risk Prediction System with Explainable AI

Predicts a person's risk of **diabetes** and **heart disease** together, and explains *why* the model made that prediction in plain English using Explainable AI (SHAP, LIME) — instead of only giving a black-box "at risk / not at risk" label.

## Why this project

Diabetes and heart disease are clinically linked (diabetic patients face 2-4x higher heart disease risk), but most prediction projects handle only one disease at a time, and rarely explain *why* a prediction was made. This project combines both diseases into a single explainable system.

## Project Phases

1. Business Understanding
2. Data Collection
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. ML Models (Logistic Regression, Random Forest, XGBoost)
7. Model Evaluation
8. Explainable AI (SHAP, LIME)
9. Business Insights

## Datasets

- **Diabetes:** Pima Indians Diabetes Dataset (UCI / Kaggle)
- **Heart Disease:** Heart Failure Prediction Dataset (Kaggle)

## Project Structure

```
data/raw/          Original, unmodified datasets
data/processed/    Cleaned and prepared data
notebooks/         Jupyter notebooks, one per phase
src/               Reusable Python code
models/            Saved trained models
reports/figures/   Charts and plots
```

## Setup

```
pip install -r requirements.txt
```
