"""
Unit and integration test suite for ClassiScope.
Verifies dataset loading, preprocessing, model pipelines, evaluation metrics,
experiments, and report generation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import ensure_dataset_exists, load_raw_data, prepare_dataset
from src.models import build_all_pipelines, build_pipeline, get_model_registry
from src.evaluation import (
    evaluate_all_models,
    extract_feature_importance,
    run_training_size_experiment,
    run_leakage_comparison_experiment,
)
from src.report_generator import generate_markdown_report


class TestClassiScopePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.csv_path = ensure_dataset_exists()
        cls.raw_df = load_raw_data(cls.csv_path)
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te, cls.num_cols, cls.cat_cols, _ = prepare_dataset(
            csv_path=cls.csv_path, test_size=0.20, random_state=42, include_period_grades=False
        )

    def test_01_dataset_integrity(self):
        """Verifies raw dataset shape and semicolon parsing."""
        self.assertEqual(len(self.raw_df), 649, "student-por.csv should contain 649 records")
        self.assertIn("G3", self.raw_df.columns)
        self.assertIn("school", self.raw_df.columns)
        self.assertEqual(self.raw_df["G3"].dtype, np.int64)

    def test_02_leakage_prevention(self):
        """Confirms that G1, G2, and G3 are strictly excluded from feature sets."""
        self.assertNotIn("G1", self.X_tr.columns, "G1 must be removed to avoid data leakage")
        self.assertNotIn("G2", self.X_tr.columns, "G2 must be removed to avoid data leakage")
        self.assertNotIn("G3", self.X_tr.columns, "G3 must be removed from input features")
        self.assertEqual(len(self.X_tr), 519)
        self.assertEqual(len(self.X_te), 130)

    def test_03_target_definition_and_class_balance(self):
        """Verifies binary target G3 >= 10."""
        pos_count = (self.raw_df["G3"] >= 10).sum()
        neg_count = (self.raw_df["G3"] < 10).sum()
        self.assertEqual(pos_count, 549)
        self.assertEqual(neg_count, 100)
        # Check stratified balance in test split
        test_pos = (self.y_te == 1).sum()
        test_neg = (self.y_te == 0).sum()
        self.assertEqual(test_pos, 110)
        self.assertEqual(test_neg, 20)

    def test_04_model_registry_contains_all_six_algorithms(self):
        """Verifies all six required models are registered."""
        reg = get_model_registry()
        required = [
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
            "Gradient Boosting"
        ]
        for name in required:
            self.assertIn(name, reg)

    def test_05_benchmark_execution_and_metrics(self):
        """Verifies that all 6 models train and produce valid bounded metrics."""
        summary_df, results_map = evaluate_all_models(
            self.X_tr, self.y_tr, self.X_te, self.y_te,
            self.num_cols, self.cat_cols, cv_folds=5, random_state=42
        )
        self.assertEqual(len(summary_df), 6)
        for _, row in summary_df.iterrows():
            self.assertTrue(0.0 <= row["Accuracy"] <= 1.0)
            self.assertTrue(0.0 <= row["F1-Score"] <= 1.0)
            self.assertTrue(0.0 <= row["CV F1 (Mean)"] <= 1.0)
            self.assertFalse(np.isnan(row["ROC-AUC"]))

    def test_06_confusion_matrix_sum(self):
        """Verifies confusion matrix sums to exactly the test set size."""
        _, results_map = evaluate_all_models(
            self.X_tr, self.y_tr, self.X_te, self.y_te,
            self.num_cols, self.cat_cols, cv_folds=5, random_state=42
        )
        for name, r in results_map.items():
            cm = r["confusion_matrix"]
            self.assertEqual(cm.sum(), len(self.y_te))
            self.assertEqual(r["tp"] + r["tn"] + r["fp"] + r["fn"], 130)

    def test_07_feature_importance_extraction(self):
        """Verifies feature attribution extraction across model families."""
        pipe_rf = build_pipeline("Random Forest", self.num_cols, self.cat_cols, 42)
        pipe_rf.fit(self.X_tr, self.y_tr)
        feat_rf = extract_feature_importance("Random Forest", pipe_rf, self.X_tr, self.y_tr, self.X_te, self.y_te, top_n=10)
        self.assertFalse(feat_rf.empty)
        self.assertLessEqual(len(feat_rf), 10)

        pipe_lr = build_pipeline("Logistic Regression", self.num_cols, self.cat_cols, 42)
        pipe_lr.fit(self.X_tr, self.y_tr)
        feat_lr = extract_feature_importance("Logistic Regression", pipe_lr, self.X_tr, self.y_tr, self.X_te, self.y_te, top_n=10)
        self.assertFalse(feat_lr.empty)

    def test_08_leakage_experiment(self):
        """Verifies that the leakage comparison demonstrates score increase with G1/G2."""
        leak_df = run_leakage_comparison_experiment(self.raw_df, random_state=42)
        self.assertEqual(len(leak_df), 6)
        # Average F1 delta must be positive
        avg_delta = leak_df["F1 Delta (%)"].mean()
        self.assertGreater(avg_delta, 0.0, "Leakage should increase F1 score")

    def test_09_report_generation(self):
        """Verifies automated report output."""
        summary_df, _ = evaluate_all_models(
            self.X_tr, self.y_tr, self.X_te, self.y_te,
            self.num_cols, self.cat_cols, cv_folds=5, random_state=42
        )
        leak_df = run_leakage_comparison_experiment(self.raw_df, random_state=42)
        dist = {"Higher Grade (>=10)": 549, "Higher Grade Ratio": 84.59, "Lower Grade (<10)": 100, "Lower Grade Ratio": 15.41}
        out_path = generate_markdown_report(summary_df, leak_df, dist, output_dir="report")
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 500)


if __name__ == "__main__":
    unittest.main()
