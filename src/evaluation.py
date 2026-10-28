"""
Evaluation and experimentation engine for ClassiScope.
Computes test metrics, stratified cross-validation distributions,
confusion matrices, ROC/PR curves, feature importances, and controlled experiments.
"""

from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from src.models import build_all_pipelines, build_pipeline, get_model_registry


def evaluate_single_pipeline(
    name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a pipeline and computes full evaluation suite:
    - Holdout test metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    - 5-fold Stratified Cross-Validation on training data: Mean and Std for F1 and Accuracy
    - Confusion Matrix components
    - ROC and PR curves
    """
    # Fit strictly on training set
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        # Fallback to decision_function calibrated if available
        if hasattr(pipeline.named_steps["classifier"], "decision_function"):
            scores = pipeline.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-scores))
        else:
            y_proba = y_pred.astype(float)
            
    # Test metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)
    except Exception:
        roc_auc = float("nan")
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        pr_prec, pr_rec = np.array([1, 0]), np.array([0, 1])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # tn: actual 0, pred 0; fp: actual 0, pred 1
    # fn: actual 1, pred 0; tp: actual 1, pred 1
    tn, fp, fn, tp = cm.ravel()
    
    # Cross-validation strictly on training data
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    try:
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    except Exception:
        cv_auc = np.array([float("nan")] * cv_folds)
        
    return {
        "model_name": name,
        "pipeline": pipeline,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_f1_macro": f1_macro,
        "test_roc_auc": roc_auc,
        "cv_f1_mean": float(np.mean(cv_f1)),
        "cv_f1_std": float(np.std(cv_f1)),
        "cv_f1_scores": cv_f1.tolist(),
        "cv_acc_mean": float(np.mean(cv_acc)),
        "cv_acc_std": float(np.std(cv_acc)),
        "cv_acc_scores": cv_acc.tolist(),
        "cv_auc_mean": float(np.mean(cv_auc)) if not np.isnan(cv_auc[0]) else float("nan"),
        "cv_auc_std": float(np.std(cv_auc)) if not np.isnan(cv_auc[0]) else float("nan"),
        "cv_auc_scores": cv_auc.tolist(),
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_prec,
        "pr_recall": pr_rec,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_test": y_test.values,
    }


def evaluate_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates all six algorithms, returns a summary metrics DataFrame and detailed results map.
    """
    pipelines = build_all_pipelines(num_cols, cat_cols, random_state=random_state)
    results_map = {}
    rows = []
    
    for name, pipe in pipelines.items():
        res = evaluate_single_pipeline(
            name, pipe, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, random_state=random_state
        )
        results_map[name] = res
        rows.append({
            "Model": name,
            "Accuracy": res["test_accuracy"],
            "Precision": res["test_precision"],
            "Recall": res["test_recall"],
            "F1-Score": res["test_f1"],
            "Macro F1": res["test_f1_macro"],
            "ROC-AUC": res["test_roc_auc"],
            "CV F1 (Mean)": res["cv_f1_mean"],
            "CV F1 (Std)": res["cv_f1_std"],
            "CV Acc (Mean)": res["cv_acc_mean"],
            "CV Acc (Std)": res["cv_acc_std"],
        })
        
    df_summary = pd.DataFrame(rows)
    return df_summary, results_map


def get_feature_names_from_pipeline(pipeline) -> List[str]:
    """
    Extracts transformed feature names from the fitted ColumnTransformer.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback names
        return [f"Feature_{i}" for i in range(100)]


def extract_feature_importance(
    name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Extracts or computes feature importance for any of the 6 models:
    - Tree/Ensemble: Mean Decrease in Impurity (.feature_importances_)
    - Logistic Regression: Normalized absolute coefficients
    - KNN / SVM: Permutation Importance computed on the test set
    """
    feature_names = get_feature_names_from_pipeline(pipeline)
    clf = pipeline.named_steps["classifier"]
    
    importances = None
    metric_label = "Relative Importance"
    
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        metric_label = "MDI (Gini / Impurity Reduction)"
    elif hasattr(clf, "coef_"):
        # Logistic Regression coefficients
        coef = clf.coef_[0]
        importances = np.abs(coef)
        metric_label = "Absolute Standardized Coefficient"
    else:
        # Non-linear / distance-based model: compute permutation importance
        perm = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=random_state,
            scoring="f1"
        )
        # Permutation importance is returned at the raw input feature level
        raw_cols = list(X_test.columns)
        perm_df = pd.DataFrame({
            "Feature": raw_cols,
            "Importance": perm.importances_mean,
            "Std": perm.importances_std,
            "Metric": "Permutation Importance (Mean F1 Drop)"
        }).sort_values(by="Importance", ascending=False).head(top_n)
        return perm_df

    if importances is not None:
        # Match lengths if needed
        min_len = min(len(feature_names), len(importances))
        feat_df = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "Importance": importances[:min_len],
            "Metric": metric_label
        }).sort_values(by="Importance", ascending=False).head(top_n)
        return feat_df
        
    return pd.DataFrame(columns=["Feature", "Importance", "Metric"])


def run_training_size_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    proportions: List[float] = [0.3, 0.45, 0.6, 0.75, 0.9, 1.0],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates how model performance evolves across training sample sizes.
    """
    records = []
    total_train = len(X_train)
    registry = get_model_registry(random_state)
    
    for prop in proportions:
        n_samples = max(int(total_train * prop), 30)
        # Stratified sub-sample of training set
        sub_X_train = X_train.iloc[:n_samples]
        sub_y_train = y_train.iloc[:n_samples]
        
        # Check that both classes exist
        if len(np.unique(sub_y_train)) < 2:
            continue
            
        for name in registry.keys():
            pipe = build_pipeline(name, num_cols, cat_cols, random_state=random_state)
            pipe.fit(sub_X_train, sub_y_train)
            
            y_pred_train = pipe.predict(sub_X_train)
            y_pred_test = pipe.predict(X_test)
            
            train_f1 = f1_score(sub_y_train, y_pred_train, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            records.append({
                "Proportion": prop,
                "Sample Count": n_samples,
                "Model": name,
                "Train F1": train_f1,
                "Test F1": test_f1,
                "Test Accuracy": test_acc
            })
            
    return pd.DataFrame(records)


def run_leakage_comparison_experiment(
    raw_df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Controlled experiment contrasting performance without prior grades (Standard Benchmark)
    versus with prior period grades (G1, G2), demonstrating the empirical impact of data leakage.
    """
    target = (raw_df["G3"] >= 10).astype(int)
    
    # Config 1: Realistic Early Warning (No G1, G2, G3)
    X_noleak = raw_df.drop(columns=["G1", "G2", "G3"])
    num_noleak = X_noleak.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_noleak = X_noleak.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Config 2: Leaked prior period grades (Includes G1, G2)
    X_leak = raw_df.drop(columns=["G3"])
    num_leak = X_leak.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_leak = X_leak.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    from sklearn.model_selection import train_test_split
    X_tr_no, X_te_no, y_tr, y_te = train_test_split(
        X_noleak, target, test_size=0.2, random_state=random_state, stratify=target
    )
    X_tr_lk, X_te_lk, _, _ = train_test_split(
        X_leak, target, test_size=0.2, random_state=random_state, stratify=target
    )
    
    records = []
    registry = get_model_registry(random_state)
    
    for name in registry.keys():
        # Evaluate without leakage
        p_no = build_pipeline(name, num_noleak, cat_noleak, random_state=random_state)
        p_no.fit(X_tr_no, y_tr)
        f1_no = f1_score(y_te, p_no.predict(X_te_no), zero_division=0)
        acc_no = accuracy_score(y_te, p_no.predict(X_te_no))
        
        # Evaluate with leakage
        p_lk = build_pipeline(name, num_leak, cat_leak, random_state=random_state)
        p_lk.fit(X_tr_lk, y_tr)
        f1_lk = f1_score(y_te, p_lk.predict(X_te_lk), zero_division=0)
        acc_lk = accuracy_score(y_te, p_lk.predict(X_te_lk))
        
        records.append({
            "Model": name,
            "F1 (Without G1/G2)": f1_no,
            "F1 (With G1/G2 Leakage)": f1_lk,
            "F1 Delta (%)": round((f1_lk - f1_no) * 100, 2),
            "Acc (Without G1/G2)": acc_no,
            "Acc (With G1/G2 Leakage)": acc_lk,
        })
        
    return pd.DataFrame(records)
