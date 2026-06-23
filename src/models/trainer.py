"""
trainer.py
----------
Trains and evaluates all classifiers for a given disease dataset.
Returns structured results including metrics, predictions, and best model.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ── Data class for results ────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_mean: float
    cv_std: float
    y_test: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    pipeline: Any  # fitted sklearn Pipeline
    confusion: np.ndarray = field(default_factory=lambda: np.array([]))
    report: str = ""


@dataclass
class DiseaseResults:
    disease: str
    results: list[ModelResult]
    best_model_name: str
    X_test: pd.DataFrame
    y_test: np.ndarray
    feature_names: list[str]


# ── Classifiers ───────────────────────────────────────────────────────────────

def _get_classifiers() -> dict:
    clfs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
    }
    if XGBOOST_AVAILABLE:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_STATE, eval_metric="logloss",
            use_label_encoder=False, verbosity=0
        )
    else:
        logger.warning("XGBoost not installed — skipping.")
    return clfs


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(
    disease: str,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    output_dir: str = "outputs",
) -> DiseaseResults:
    """
    Train all classifiers on the given dataset.

    Steps
    -----
    1. Train/test split (stratified)
    2. For each classifier:
       a. Build a Pipeline(preprocessor → classifier)
       b. Fit on train set
       c. Evaluate on test set
       d. 5-fold cross-validation on full dataset
    3. Select best model by ROC-AUC
    4. Save best model to outputs/models/

    Returns
    -------
    DiseaseResults dataclass
    """
    logger.info("=" * 60)
    logger.info("Training models for: %s", disease.upper())
    logger.info("Dataset shape: %s | Class balance: %.1f%% positive",
                X.shape, y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    classifiers = _get_classifiers()
    results: list[ModelResult] = []

    for name, clf in classifiers.items():
        logger.info("  Training: %s ...", name)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            logger.error("  %s failed to fit: %s", name, e)
            continue

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

        # Cross-validation on the full dataset (refitting inside CV)
        cv_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        cv_scores = cross_val_score(cv_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

        result = ModelResult(
            name=name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=auc,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            y_test=np.array(y_test),
            y_pred=y_pred,
            y_prob=y_prob,
            pipeline=pipe,
            confusion=cm,
            report=report,
        )
        results.append(result)

        logger.info(
            "  %-22s | Acc: %.3f | Prec: %.3f | Rec: %.3f | F1: %.3f | AUC: %.3f | CV: %.3f±%.3f",
            name, acc, prec, rec, f1, auc, cv_scores.mean(), cv_scores.std()
        )

    # Select best model
    best = max(results, key=lambda r: r.roc_auc)
    logger.info("Best model for %s: %s (ROC-AUC: %.4f)", disease, best.name, best.roc_auc)

    # Save best model
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{disease}_best_model.pkl")
    joblib.dump(best.pipeline, model_path)
    logger.info("Best model saved to %s", model_path)

    # Save classification report
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    _save_report(disease, results, best.name, report_dir)

    return DiseaseResults(
        disease=disease,
        results=results,
        best_model_name=best.name,
        X_test=X_test,
        y_test=np.array(y_test),
        feature_names=list(X.columns),
    )


def _save_report(disease: str, results: list[ModelResult], best_name: str, report_dir: str):
    lines = [
        f"Disease Prediction Report — {disease.upper().replace('_', ' ')}",
        "=" * 60,
        f"{'Model':<25} {'Accuracy':>8} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'ROC-AUC':>9} {'CV Mean':>9} {'CV Std':>8}",
        "-" * 90,
    ]
    for r in results:
        marker = " ★" if r.name == best_name else ""
        lines.append(
            f"{r.name:<25} {r.accuracy:>8.4f} {r.precision:>10.4f} {r.recall:>8.4f} "
            f"{r.f1:>8.4f} {r.roc_auc:>9.4f} {r.cv_mean:>9.4f} {r.cv_std:>8.4f}{marker}"
        )
    lines += ["-" * 90, "", f"Best Model: {best_name}", ""]

    best_result = next(r for r in results if r.name == best_name)
    lines += ["Classification Report (Best Model):", best_result.report]

    path = os.path.join(report_dir, f"{disease}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report saved to %s", path)
