"""
Standalone CLI entry point to run full model evaluation and generate academic reports.
Usage:
    python run_experiments.py
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import prepare_dataset, ensure_dataset_exists
from src.evaluation import evaluate_all_models, run_leakage_comparison_experiment
from src.report_generator import generate_markdown_report


def main():
    print("=" * 70)
    print("ClassiScope: Classification Algorithm Performance Benchmark")
    print("Dataset: UCI Student Performance (Portuguese - student-por.csv)")
    print("=" * 70)

    # 1. Ensure dataset exists
    print("\n[1/4] Checking and preparing dataset...")
    csv_path = ensure_dataset_exists()
    print(f"Dataset path: {csv_path}")

    X_train, X_test, y_train, y_test, num_cols, cat_cols, raw_df = prepare_dataset(
        csv_path=csv_path,
        test_size=0.20,
        random_state=42,
        include_period_grades=False
    )
    
    n_pos = int((raw_df["G3"] >= 10).sum())
    n_neg = int((raw_df["G3"] < 10).sum())
    class_dist = {
        "Total": len(raw_df),
        "Higher Grade (>=10)": n_pos,
        "Higher Grade Ratio": round((n_pos / len(raw_df)) * 100, 2),
        "Lower Grade (<10)": n_neg,
        "Lower Grade Ratio": round((n_neg / len(raw_df)) * 100, 2),
    }
    
    print(f"Total instances: {len(raw_df)}")
    print(f"Train split: {len(X_train)} | Test split: {len(X_test)}")
    print(f"Numerical features: {len(num_cols)} | Categorical features: {len(cat_cols)}")
    print(f"Class breakdown: Higher Grade (>=10): {n_pos} ({class_dist['Higher Grade Ratio']}%) | Lower Grade (<10): {n_neg} ({class_dist['Lower Grade Ratio']}%)")

    # 2. Train and evaluate all 6 models
    print("\n[2/4] Training and evaluating all 6 algorithms...")
    summary_df, results_map = evaluate_all_models(
        X_train, y_train, X_test, y_test, num_cols, cat_cols, cv_folds=5, random_state=42
    )

    print("\nBenchmark Results Table:")
    print("-" * 100)
    print(summary_df.to_string(index=False))
    print("-" * 100)

    # 3. Data Leakage experiment
    print("\n[3/4] Running data leakage ablation experiment (without G1/G2 vs with G1/G2)...")
    leakage_df = run_leakage_comparison_experiment(raw_df, random_state=42)
    print(leakage_df.to_string(index=False))

    # 4. Generate report
    print("\n[4/4] Generating academic summary report...")
    report_file = generate_markdown_report(summary_df, leakage_df, class_dist, output_dir="report")
    print(f"Report generated successfully: {report_file}")
    print("JSON results saved to: report/results.json")
    print("\n[SUCCESS] All 6 models verified and pipeline completed without errors.")


if __name__ == "__main__":
    main()
