"""
data_loader.py
--------------
Loads Heart Disease, Diabetes, and Breast Cancer datasets.
All datasets are fetched automatically — no manual download needed.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger(__name__)


# ── Heart Disease ─────────────────────────────────────────────────────────────

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def load_heart_disease(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the Cleveland Heart Disease dataset.
    Downloads to data_dir if not already present.
    Target: 0 = no disease, 1 = disease (original labels 1-4 collapsed to 1).
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "heart_disease.csv")

    if os.path.exists(cache_path):
        logger.info("Heart Disease: loading from cache %s", cache_path)
        df = pd.read_csv(cache_path)
    else:
        logger.info("Heart Disease: downloading from UCI...")
        try:
            df = pd.read_csv(HEART_URL, header=None, names=HEART_COLUMNS,
                             na_values="?")
            df.to_csv(cache_path, index=False)
            logger.info("Heart Disease: saved to %s", cache_path)
        except Exception as e:
            logger.warning("Download failed (%s). Using sklearn heart dataset fallback.", e)
            df = _heart_fallback()

    # Collapse multi-class target to binary
    df["target"] = (df["target"] > 0).astype(int)
    logger.info("Heart Disease loaded: %d rows, %d cols. Positive rate: %.1f%%",
                len(df), df.shape[1], df["target"].mean() * 100)
    return df


def _heart_fallback() -> pd.DataFrame:
    """Fallback: generate a heart-disease-like dataset from sklearn's fetch."""
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("heart-c", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        df.columns = [c.lower().replace("-", "_") for c in df.columns]
        # Rename target column
        target_col = [c for c in df.columns if "class" in c or "target" in c]
        if target_col:
            df = df.rename(columns={target_col[0]: "target"})
            df["target"] = (df["target"].astype(str) != "tested_negative").astype(int)
        return df
    except Exception:
        # Last resort: synthetic heart data
        logger.warning("All heart data sources failed. Using synthetic data.")
        return _synthetic_heart()


def _synthetic_heart() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 303
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 565, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.integers(0, 4, n),
        "target": rng.integers(0, 2, n),
    })
    return df


# ── Diabetes ──────────────────────────────────────────────────────────────────

DIABETES_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
]

DIABETES_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)


def load_diabetes(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes dataset.
    Target column: 'outcome' — 0 = no diabetes, 1 = diabetes.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "diabetes.csv")

    if os.path.exists(cache_path):
        logger.info("Diabetes: loading from cache %s", cache_path)
        df = pd.read_csv(cache_path)
    else:
        logger.info("Diabetes: downloading...")
        try:
            df = pd.read_csv(DIABETES_URL, header=None, names=DIABETES_COLUMNS)
            df.to_csv(cache_path, index=False)
        except Exception as e:
            logger.warning("Diabetes download failed (%s). Using sklearn fallback.", e)
            df = _diabetes_fallback()

    # Replace impossible zero values with NaN for relevant columns
    zero_invalid = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    df[zero_invalid] = df[zero_invalid].replace(0, np.nan)

    logger.info("Diabetes loaded: %d rows, %d cols. Positive rate: %.1f%%",
                len(df), df.shape[1], df["outcome"].mean() * 100)
    return df


def _diabetes_fallback() -> pd.DataFrame:
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        df.columns = DIABETES_COLUMNS[: len(df.columns) - 1] + ["outcome"]
        df["outcome"] = (df["outcome"].astype(str) == "tested_positive").astype(int)
        return df
    except Exception:
        logger.warning("All diabetes sources failed. Using synthetic data.")
        return _synthetic_diabetes()


def _synthetic_diabetes() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 768
    return pd.DataFrame({
        "pregnancies": rng.integers(0, 18, n),
        "glucose": rng.integers(44, 199, n, dtype=int).astype(float),
        "blood_pressure": rng.integers(24, 122, n, dtype=int).astype(float),
        "skin_thickness": rng.integers(0, 99, n, dtype=int).astype(float),
        "insulin": rng.integers(0, 846, n, dtype=int).astype(float),
        "bmi": rng.uniform(18, 68, n).round(1),
        "diabetes_pedigree": rng.uniform(0.07, 2.42, n).round(3),
        "age": rng.integers(21, 81, n),
        "outcome": rng.integers(0, 2, n),
    })


# ── Breast Cancer ─────────────────────────────────────────────────────────────

def load_breast_cancer_data() -> pd.DataFrame:
    """
    Load the Breast Cancer Wisconsin dataset from sklearn.
    Target: 0 = malignant, 1 = benign.
    """
    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["target"] = raw.target
    logger.info("Breast Cancer loaded: %d rows, %d cols. Benign rate: %.1f%%",
                len(df), df.shape[1], df["target"].mean() * 100)
    return df


# ── Dispatcher ────────────────────────────────────────────────────────────────

LOADERS = {
    "heart": load_heart_disease,
    "diabetes": load_diabetes,
    "breast_cancer": load_breast_cancer_data,
}

TARGET_COLUMNS = {
    "heart": "target",
    "diabetes": "outcome",
    "breast_cancer": "target",
}


def load_dataset(disease: str, data_dir: str = "data") -> tuple[pd.DataFrame, str]:
    """
    Load a dataset by disease name.

    Parameters
    ----------
    disease : str
        One of 'heart', 'diabetes', 'breast_cancer'.
    data_dir : str
        Directory to cache downloaded data.

    Returns
    -------
    df : pd.DataFrame
    target_col : str
    """
    if disease not in LOADERS:
        raise ValueError(f"Unknown disease '{disease}'. Choose from: {list(LOADERS)}")

    loader = LOADERS[disease]
    if disease in ("heart", "diabetes"):
        df = loader(data_dir=data_dir)
    else:
        df = loader()

    target_col = TARGET_COLUMNS[disease]
    return df, target_col
