"""
tests/test_ingestion.py
-----------------------
Tests for ModelInput dataclass and load_model_input() validation.

Run with:  python -m unittest tests.test_ingestion -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from core.ingestion import load_model_input, IngestionError, ModelInput


# ── Shared setup ─────────────────────────────────────────────────────────────

def _make_clf_bundle():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series(y, name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def _make_reg_bundle():
    X, y = make_regression(n_samples=200, n_features=4, random_state=0)
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    y = pd.Series(y, name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ── Test 1: Valid inputs ──────────────────────────────────────────────────────

class TestValidInput(unittest.TestCase):

    def setUp(self):
        self.clf = _make_clf_bundle()
        self.reg = _make_reg_bundle()

    def test_returns_model_input_instance(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="classification")
        self.assertIsInstance(mi, ModelInput)

    def test_model_name_auto_inferred(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="classification")
        self.assertEqual(mi.model_name, "LogisticRegression")

    def test_custom_model_name(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="classification", model_name="MyModel")
        self.assertEqual(mi.model_name, "MyModel")

    def test_feature_names_auto_inferred(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="classification")
        self.assertEqual(mi.feature_names, [f"feat_{i}" for i in range(5)])

    def test_row_counts_correct(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="classification")
        self.assertEqual(mi.n_train, len(X_train))
        self.assertEqual(mi.n_test,  len(X_test))

    def test_classes_property_classification(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="classification")
        self.assertEqual(set(mi.classes), {0, 1})

    def test_classes_property_regression_is_none(self):
        model, X_train, X_test, y_train, y_test = self.reg
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="regression")
        self.assertIsNone(mi.classes)

    def test_numpy_arrays_accepted(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model,
                              X_train.values, X_test.values,
                              y_train.values, y_test.values,
                              task_type="classification")
        self.assertIsInstance(mi.X_train, pd.DataFrame)
        self.assertIsInstance(mi.y_train, pd.Series)

    def test_domain_and_task_normalised_lowercase(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="Classification", domain="HEALTHCARE")
        self.assertEqual(mi.domain,    "healthcare")
        self.assertEqual(mi.task_type, "classification")

    def test_model_hash_is_sha256(self):
        model, X_train, X_test, y_train, y_test = self.clf
        mi = load_model_input(model, X_train, X_test, y_train, y_test,
                              task_type="classification")
        self.assertIsInstance(mi.model_hash, str)
        self.assertEqual(len(mi.model_hash), 64)


# ── Test 2: Invalid task_type and domain ─────────────────────────────────────

class TestInvalidTaskAndDomain(unittest.TestCase):

    def setUp(self):
        self.clf = _make_clf_bundle()

    def test_invalid_task_type_raises(self):
        model, X_train, X_test, y_train, y_test = self.clf
        with self.assertRaises(IngestionError):
            load_model_input(model, X_train, X_test, y_train, y_test,
                             task_type="clustering")

    def test_invalid_domain_raises(self):
        model, X_train, X_test, y_train, y_test = self.clf
        with self.assertRaises(IngestionError):
            load_model_input(model, X_train, X_test, y_train, y_test,
                             task_type="classification", domain="sports")

    def test_model_without_predict_raises(self):
        _, X_train, X_test, y_train, y_test = self.clf
        class FakeModel:
            pass
        with self.assertRaises(IngestionError):
            load_model_input(FakeModel(), X_train, X_test, y_train, y_test,
                             task_type="classification")


# ── Test 3: Shape and column validation ──────────────────────────────────────

class TestShapeValidation(unittest.TestCase):

    def setUp(self):
        self.clf = _make_clf_bundle()

    def test_column_mismatch_raises(self):
        model, X_train, X_test, y_train, y_test = self.clf
        X_test_bad = X_test.rename(columns={"feat_0": "wrong_col"})
        with self.assertRaises(IngestionError):
            load_model_input(model, X_train, X_test_bad, y_train, y_test,
                             task_type="classification")

    def test_y_length_mismatch_raises(self):
        model, X_train, X_test, y_train, y_test = self.clf
        y_test_bad = y_test.iloc[:-3]
        with self.assertRaises(IngestionError):
            load_model_input(model, X_train, X_test, y_train, y_test_bad,
                             task_type="classification")

    def test_empty_dataframe_raises(self):
        model, X_train, X_test, y_train, y_test = self.clf
        with self.assertRaises(IngestionError):
            load_model_input(model, pd.DataFrame(), X_test, y_train, y_test,
                             task_type="classification")

    def test_too_few_train_rows_raises(self):
        model, X_train, X_test, y_train, y_test = self.clf
        tiny_X = X_train.iloc[:3]
        tiny_y = y_train.iloc[:3]
        with self.assertRaises(IngestionError):
            load_model_input(model, tiny_X, X_test, tiny_y, y_test,
                             task_type="classification")

    def test_wrong_type_for_X_raises(self):
        model, X_train, X_test, y_train, y_test = self.clf
        with self.assertRaises(IngestionError):
            load_model_input(model, "not_a_dataframe", X_test, y_train, y_test,
                             task_type="classification")


if __name__ == "__main__":
    unittest.main(verbosity=2)