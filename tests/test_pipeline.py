"""
test_pipeline.py
----------------
Unit tests for all pipeline stages.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.data_loader import load_dataset, DISEASES
from src.features.feature_engineering import prepare_features


# ── Data Loading ──────────────────────────────────────────────────────────────

class TestDataLoader:

    def test_breast_cancer_loads(self):
        df, target_col = load_dataset("breast_cancer")
        assert isinstance(df, pd.DataFrame)
        assert target_col in df.columns
        assert len(df) > 100
        assert df[target_col].nunique() == 2

    def test_diabetes_loads(self):
        df, target_col = load_dataset("diabetes")
        assert "outcome" in df.columns
        assert df.shape[1] >= 9

    def test_heart_loads(self):
        df, target_col = load_dataset("heart")
        assert "target" in df.columns

    def test_invalid_disease_raises(self):
        with pytest.raises(ValueError):
            load_dataset("flu")


# ── Feature Engineering ───────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_breast_cancer_features(self):
        df, target_col = load_dataset("breast_cancer")
        X, y, preprocessor = prepare_features("breast_cancer", df, target_col)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 30  # added engineered features
        assert set(y.unique()).issubset({0, 1})

    def test_diabetes_features(self):
        df, target_col = load_dataset("diabetes")
        X, y, preprocessor = prepare_features("diabetes", df, target_col)
        assert "glucose_insulin_ratio" in X.columns
        assert "bmi_category" in X.columns
        assert "high_glucose" in X.columns

    def test_heart_features(self):
        df, target_col = load_dataset("heart")
        X, y, preprocessor = prepare_features("heart", df, target_col)
        assert "age_group" in X.columns
        assert "chol_bp_ratio" in X.columns

    def test_preprocessor_transforms(self):
        """Check that preprocessor produces valid numpy arrays."""
        from sklearn.model_selection import train_test_split
        df, target_col = load_dataset("breast_cancer")
        X, y, preprocessor = prepare_features("breast_cancer", df, target_col)
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        Xt = preprocessor.fit_transform(X_train)
        assert not np.any(np.isnan(Xt))
        assert Xt.shape[0] == X_train.shape[0]


# ── Trainer (smoke test) ──────────────────────────────────────────────────────

class TestTrainer:

    def test_breast_cancer_trains(self, tmp_path):
        from src.models.trainer import train_and_evaluate
        df, target_col = load_dataset("breast_cancer")
        X, y, preprocessor = prepare_features("breast_cancer", df, target_col)
        result = train_and_evaluate(
            "breast_cancer", X, y, preprocessor, output_dir=str(tmp_path)
        )
        assert len(result.results) >= 4
        assert result.best_model_name in [r.name for r in result.results]
        best = next(r for r in result.results if r.name == result.best_model_name)
        assert best.roc_auc > 0.5  # Must beat random

    def test_diabetes_trains(self, tmp_path):
        from src.models.trainer import train_and_evaluate
        df, target_col = load_dataset("diabetes")
        X, y, preprocessor = prepare_features("diabetes", df, target_col)
        result = train_and_evaluate(
            "diabetes", X, y, preprocessor, output_dir=str(tmp_path)
        )
        best = next(r for r in result.results if r.name == result.best_model_name)
        assert best.roc_auc > 0.5
