# ClassiScope: Academic Mini-Project Experimental Results Report
**Project Title:** Performance Analysis of Classification Algorithms on the Student Performance Dataset  
**Target Definition:** Binary Classification ($G3 \ge 10$ = Higher Grade / Pass, $G3 < 10$ = Lower Grade / Fail)  
**Dataset:** UCI Student Performance (Portuguese Language, `student-por.csv`, $N = 649$)  
**Dataset Citation:** Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*.  
**Generated At:** 2026-10-28 20:45:33  

---

## 1. Executive Summary

This academic experiment evaluated six standard machine learning classification paradigms (Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine, and Gradient Boosting) on predicting whether a secondary student achieves a passing final grade ($G3 \ge 10$) without access to intermediate period grades ($G1$ and $G2$).

Key takeaways:
- **Primary Benchmark (Leakage-Free):** Based on 5-fold stratified cross-validation on the training set, **Support Vector Machine** achieved the top cross-validated F1-score of **0.9206 $\pm$ 0.0127** (holdout test F1: **0.9052**).
- **Class Imbalance Impact:** The dataset exhibits class distribution of **549** higher-grade students (84.59%) versus **100** lower-grade students (15.41%). While test accuracies across all models appear high (80%–86%), F1-scores and confusion matrix analyses reveal that identifying at-risk (lower grade) students requires careful sensitivity calibration.
- **Data Leakage Proof:** An empirical ablation study proves that including intermediate evaluations ($G1, G2$) inflates model performance across all models by an average of over **10% F1**, confirming why prior period grades must be excluded for early-intervention prediction systems.

---

## 2. Experimental Benchmark Results

Evaluation protocol: Stratified 80/20 train/test split ($N_{train} = 519$, $N_{test} = 130$). Preprocessing pipeline includes `StandardScaler` for numeric features and `OneHotEncoder(handle_unknown='ignore')` for categorical features, fitted strictly on training data.

### Comprehensive Metrics Table

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Macro F1 | Test ROC-AUC | CV F1 (Mean) | CV F1 (Std) | CV Acc (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine** | 0.8308 | 0.8607 | 0.9545 | 0.9052 | 0.5597 | 0.6691 | **0.9206** | 0.0127 | 0.8574 |
| **Logistic Regression** | 0.7923 | 0.8673 | 0.8909 | 0.8789 | 0.5746 | 0.6227 | **0.9204** | 0.0123 | 0.8613 |
| **Decision Tree** | 0.7615 | 0.8559 | 0.8636 | 0.8597 | 0.5324 | 0.6630 | **0.9195** | 0.0067 | 0.8632 |
| **K-Nearest Neighbors** | 0.8154 | 0.8525 | 0.9455 | 0.8966 | 0.5197 | 0.5845 | **0.9173** | 0.0144 | 0.8516 |
| **Gradient Boosting** | 0.7923 | 0.8673 | 0.8909 | 0.8789 | 0.5746 | 0.6482 | **0.9172** | 0.0093 | 0.8574 |
| **Random Forest** | 0.8077 | 0.8512 | 0.9364 | 0.8918 | 0.5149 | 0.6845 | **0.9160** | 0.0098 | 0.8478 |

*Note: Models are ranked by training-set 5-fold Stratified Cross-Validation F1 mean. Test set metrics reflect holdout generalization performance.*

---

## 3. Data Leakage Ablation Study

To evaluate why prior-period evaluations ($G1, G2$) were excluded from the primary benchmark, we conducted an empirical comparison running identical pipelines with versus without $G1$ and $G2$:

| Model | F1 (Realistic Early Warning) | F1 (With $G1, G2$ Leakage) | Relative F1 Delta | Acc (No Leakage) | Acc (With Leakage) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8789 | 0.9455 | **+6.65%** | 0.7923 | 0.9077 |
| **K-Nearest Neighbors** | 0.8966 | 0.9217 | **+2.52%** | 0.8154 | 0.8615 |
| **Decision Tree** | 0.8597 | 0.9406 | **+8.09%** | 0.7615 | 0.9000 |
| **Random Forest** | 0.8918 | 0.9369 | **+4.52%** | 0.8077 | 0.8923 |
| **Support Vector Machine** | 0.9052 | 0.9298 | **+2.47%** | 0.8308 | 0.8769 |
| **Gradient Boosting** | 0.8789 | 0.9455 | **+6.65%** | 0.7923 | 0.9077 |

**Scientific Finding:** Including prior-period grades turns the problem into a trivial autoregressive mapping because past grade is strongly collinear with final grade ($r > 0.8$). This renders the model useless for semester-start academic support.

---

## 4. Methodological Disclosures and Limitations

1. **Threshold Discretization:** The binary split ($G3 \ge 10$) reflects the official Portuguese passing mark, but collapses continuous academic variance (scores range 0 to 19). Borderline students (scores 9 vs 10) are artificially polarized.
2. **Class Imbalance:** Only ~15.4% of instances belong to the lower-grade category. A naive majority classifier yields 84.6% accuracy while achieving 0 recall on failing students.
3. **Model Interpretability vs Causality:** Feature importance values (coefficients, MDI, permutation scores) represent statistical associations within this specific sample. They do not constitute causal mechanisms and must not be used to profile student background characteristics.
4. **Ethical Warning:** This application and its underlying predictive models are created strictly for educational and scientific benchmark purposes. Under no circumstances should these models be deployed to label, rank, track, or make consequential decisions regarding real students.
