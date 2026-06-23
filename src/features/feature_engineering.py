"""
feature_engineering.py
-----------------------
Preprocessing and feature engineering for all three disease datasets.
Each disease has a custom pipeline reflecting its domain knowledge.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def _numeric_pipeline(strategy: str = "median") -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy=strategy)),
        ("scaler", StandardScaler()),
    ])


# ── Heart Disease ─────────────────────────────────────────────────────────────

HEART_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
HEART_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def engineer_heart(df: pd.DataFrame, target_col: str = "target"):
    """
    Heart Disease feature engineering.

    New features:
    - age_group: discretized age bins
    - chol_trestbps_ratio: cholesterol / resting blood pressure
    - high_risk_combo: thalach < 120 AND oldpeak > 2
    """
    df = df.copy()

    # Drop rows with too many missing values (>50% missing)
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold).reset_index(drop=True)

    # Feature: age group
    df["age_group"] = pd.cut(df["age"], bins=[0, 40, 55, 70, 120],
                              labels=[0, 1, 2, 3]).astype(float)

    # Feature: cholesterol / blood pressure ratio
    df["chol_bp_ratio"] = df["chol"] / df["trestbps"].replace(0, np.nan)

    # Feature: high risk combination
    df["high_risk_combo"] = (
        (df["thalach"] < 120) & (df["oldpeak"] > 2)
    ).astype(int)

    X, y = _split_features_target(df, target_col)

    numeric_cols = [c for c in X.columns if c not in HEART_CATEGORICAL]
    categorical_cols = [c for c in HEART_CATEGORICAL if c in X.columns]

    preprocessor = ColumnTransformer([
        ("num", _numeric_pipeline("median"), numeric_cols),
        ("cat", _numeric_pipeline("most_frequent"), categorical_cols),
    ], remainder="drop")

    logger.info("Heart Disease: %d features after engineering", X.shape[1])
    return X, y, preprocessor


# ── Diabetes ──────────────────────────────────────────────────────────────────

DIABETES_NUMERIC = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age"
]


def engineer_diabetes(df: pd.DataFrame, target_col: str = "outcome"):
    """
    Diabetes feature engineering.

    New features:
    - bmi_category: underweight / normal / overweight / obese
    - glucose_insulin_ratio: glucose / insulin (key metabolic indicator)
    - age_bmi_interaction: age × bmi
    """
    df = df.copy()

    # Feature: BMI category
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # Feature: glucose/insulin ratio (proxy for insulin resistance)
    df["glucose_insulin_ratio"] = df["glucose"] / df["insulin"].replace(0, np.nan)

    # Feature: age × bmi interaction
    df["age_bmi"] = df["age"] * df["bmi"]

    # Feature: high glucose flag
    df["high_glucose"] = (df["glucose"] > 140).astype(int)

    X, y = _split_features_target(df, target_col)

    numeric_cols = [c for c in X.columns]  # all numeric after engineering

    preprocessor = ColumnTransformer([
        ("num", _numeric_pipeline("median"), numeric_cols),
    ], remainder="drop")

    logger.info("Diabetes: %d features after engineering", X.shape[1])
    return X, y, preprocessor


# ── Breast Cancer ─────────────────────────────────────────────────────────────

def engineer_breast_cancer(df: pd.DataFrame, target_col: str = "target"):
    """
    Breast Cancer feature engineering.
    The dataset is already clean and numeric (30 features).
    We add a few composite features.

    New features:
    - mean_area_perimeter_ratio
    - worst_area_perimeter_ratio
    - mean_texture_smoothness_interaction
    """
    df = df.copy()

    # Composite: mean area / perimeter ratio
    if "mean area" in df.columns and "mean perimeter" in df.columns:
        df["mean_area_perim_ratio"] = df["mean area"] / (df["mean perimeter"] + 1e-6)

    # Composite: worst area / worst perimeter
    if "worst area" in df.columns and "worst perimeter" in df.columns:
        df["worst_area_perim_ratio"] = df["worst area"] / (df["worst perimeter"] + 1e-6)

    # Interaction: mean texture × mean smoothness
    if "mean texture" in df.columns and "mean smoothness" in df.columns:
        df["texture_smoothness"] = df["mean texture"] * df["mean smoothness"]

    X, y = _split_features_target(df, target_col)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), list(X.columns)),
    ], remainder="drop")

    logger.info("Breast Cancer: %d features after engineering", X.shape[1])
    return X, y, preprocessor


# ── Dispatcher ────────────────────────────────────────────────────────────────

ENGINEERS = {
    "heart": engineer_heart,
    "diabetes": engineer_diabetes,
    "breast_cancer": engineer_breast_cancer,
}


def prepare_features(disease: str, df: pd.DataFrame, target_col: str):
    """
    Run disease-specific feature engineering.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    preprocessor : sklearn ColumnTransformer
    """
    if disease not in ENGINEERS:
        raise ValueError(f"No feature engineer for disease '{disease}'")
    return ENGINEERS[disease](df, target_col)
