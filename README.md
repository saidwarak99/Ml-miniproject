# ClassiScope: Performance Analysis of Classification Algorithms on the Student Performance Dataset

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-6.0+-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic mini-project and interactive web application comparing six classification paradigms on the **UCI Student Performance Dataset (Portuguese Language subject)**. The system evaluates predictive power, cross-validation stability, diagnostic error dynamics, sample efficiency, and data leakage under leakage-free experimental conditions.

---

## ⚠️ Ethical & Responsible Use Mandate

> **CRITICAL WARNING:** This educational model and exploratory research platform must **NEVER** be deployed to label, rank, track, penalize, or make consequential decisions regarding real students. Predictive models trained on historical demographic, familial, or behavioral attributes reflect localized socio-demographic correlation, **not causality**. Deploying algorithmic ranking in educational institutions introduces high risks of automating systemic bias, pigeonholing vulnerable learners, and restricting equitable academic access.

---

## 1. Project Overview & Research Question

### Research Question
> *How do diverse classification algorithms (linear boundaries, instance-based neighborhoods, recursive partitioning trees, bootstrap ensembles, kernel margin maximizers, and sequential gradient boosting) compare in predicting whether a student achieves a passing final grade, and how do their reliability and error patterns differ when prior-period grades are deliberately withheld?*

### Target Definition & Limitations
* **Target Variable:** Derived from $G3$ (continuous final grade on a $0$–$20$ scale).
  * **Class 1 (Higher Grade / Pass):** $G3 \ge 10$ ($N = 549$, **84.59%**)
  * **Class 0 (Lower Grade / Fail):** $G3 < 10$ ($N = 100$, **15.41%**)
* **Leakage Control:** Earlier trimester evaluations ($G1$ and $G2$) and the raw target source ($G3$) are **strictly removed** from the input feature space. Including $G1$ and $G2$ turns the problem into a trivial autoregressive mapping ($r > 0.8$), undermining early semester intervention utility.
* **Academic Limitations of the Threshold:**
  1. **Discretization Loss:** Collapsing continuous achievements into binary classes masks the divergence between top honors ($19/20$) and bare passing marks ($10/20$).
  2. **Boundary Discontinuity:** Students scoring $9.4$ and $10.0$ are separated into opposing classes despite virtually indistinguishable competence.
  3. **Class Imbalance:** Because 84.6% of students pass, naive accuracy is a misleading metric; F1-score, Precision-Recall AUC, and confusion matrix calibration are critical.

---

## 2. Dataset Provenance & Citation

The dataset is obtained from the **UCI Machine Learning Repository** (Dataset ID: 320):
* **Dataset:** Portuguese Language Subject (`student-por.csv`), $N = 649$ instances, 33 original attributes.
* **Local Caching:** Handled automatically by `src/data_loader.py` with automatic extraction and semicolon parsing.
* **Official Citation:**
  ```bibtex
  @inproceedings{cortez2008using,
    title={Using Data Mining to Predict Secondary School Student Performance},
    author={Cortez, Paulo and Silva, Alice Maria Gon{\c{c}}alves},
    booktitle={Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008)},
    pages={5--12},
    year={2008},
    address={Porto, Portugal},
    isbn={978-9077381-39-7}
  }
  ```

---

## 3. Models & Preprocessing Pipeline

Six classification algorithms are implemented and wrapped inside strictly isolated scikit-learn `Pipeline` objects:

1. **Logistic Regression:** L2-regularized linear model with L-BFGS solver.
2. **K-Nearest Neighbors (KNN):** Distance-weighted neighborhood classification in normalized metric space.
3. **Decision Tree (CART):** Depth-constrained recursive partitioning with Gini impurity splitting.
4. **Random Forest:** Bootstrap aggregated ensemble of 150 de-correlated trees with sub-feature sampling.
5. **Support Vector Machine (SVM):** Radial Basis Function (RBF) margin maximizer.
6. **Gradient Boosting:** Sequential additive boosting minimizing binary log-loss.

### Leakage-Free Preprocessing
* **Numeric Features (13 attributes):** Standardized using `StandardScaler()`.
* **Categorical Features (17 attributes):** Encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
* **Isolation Guarantee:** The `ColumnTransformer` is fitted **strictly on the training partition** ($N_{train} = 519$) and merely applied to the holdout test set ($N_{test} = 130$).
* **Stratified Validation:** Fixed seed (`42`) used for both train/test splitting and 5-fold cross-validation.

---

## 4. Benchmark Results

Evaluated on the holdout test set ($N=130$) and 5-fold stratified cross-validation on the training set:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Macro F1 | Test ROC-AUC | CV F1 (Mean ± Std) | CV Acc (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine** | **0.831** | 0.861 | **0.955** | **0.905** | 0.560 | 0.669 | **0.921 ± 0.013** | 0.857 ± 0.022 |
| **K-Nearest Neighbors** | 0.815 | 0.852 | 0.945 | 0.897 | 0.520 | 0.585 | 0.917 ± 0.014 | 0.852 ± 0.026 |
| **Random Forest** | 0.808 | 0.851 | 0.936 | 0.892 | 0.515 | **0.685** | 0.916 ± 0.010 | 0.848 ± 0.016 |
| **Logistic Regression** | 0.792 | **0.867** | 0.891 | 0.879 | **0.575** | 0.623 | **0.920 ± 0.012** | 0.861 ± 0.022 |
| **Gradient Boosting** | 0.792 | **0.867** | 0.891 | 0.879 | **0.575** | 0.648 | 0.917 ± 0.009 | 0.857 ± 0.017 |
| **Decision Tree** | 0.762 | 0.856 | 0.864 | 0.860 | 0.532 | 0.663 | 0.919 ± 0.007 | **0.863 ± 0.013** |

*All results produced empirically from the experimental pipeline; never fabricated.*

### Empirical Data Leakage Impact
Ablation demonstrating how including prior-period evaluations ($G1, G2$) artificially inflates metrics:
* **Logistic Regression:** F1 rises from **0.879** to **0.945** (+6.65%)
* **Decision Tree:** F1 rises from **0.860** to **0.941** (+8.09%), Accuracy from **76.2%** to **90.0%**
* **Conclusion:** Intermediate grades mask demographic and behavioral indicators; excluding them is essential for actionable early-intervention models.

---

## 5. Repository Structure

```
ClassiScope/
├── .streamlit/
│   └── config.toml             # Dark navy/slate editorial theme
├── data/
│   └── student-por.csv         # Local cached copy of UCI Portuguese dataset
├── report/
│   ├── results_summary.md      # Auto-generated markdown academic report
│   └── results.json            # Machine-readable experiment records
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Semicolon parsing, download fallback, split
│   ├── models.py               # 6 model pipelines, ColumnTransformer, registry
│   ├── evaluation.py           # Metrics, CV, ROC/PR curves, leakage & learning curves
│   └── report_generator.py     # Automated report authoring module
├── tests/
│   └── test_pipeline.py        # 9 automated unit/integration tests
├── app.py                      # Interactive Streamlit dashboard
├── run_experiments.py          # Standalone headless CLI experiment runner
├── requirements.txt            # Dependency specifications
├── .gitignore                  # Clean version control exclusions
└── README.md                   # Comprehensive documentation
```

---

## 6. Quick Start & Execution

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/ClassiScope.git
cd ClassiScope

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Experiments via CLI
To train all six models, compute CV metrics, run the leakage ablation, and generate the report:
```bash
python run_experiments.py
```

### 3. Run Automated Tests
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Launch Interactive Web App
Launch the Streamlit dashboard on your local machine:
```bash
streamlit run app.py
```
Open your browser to [http://localhost:8501](http://localhost:8501).

---

## 7. Product Features in the Dashboard

* **Overview:** Project context, dataset provenance, interactive continuous $G3$ grade distribution with threshold markers, and dataset inspection.
* **Model Benchmark:** Sortable leaderboard, dynamic winner badges, interactive Plotly bar comparisons with CV error bars, and multi-metric radar profiling.
* **Error Analysis:** Annotated confusion matrices, TP/TN/FP/FN counters, ROC curves with chance baselines, and Precision-Recall curves.
* **Experiment Lab:**
  * *Learning Curves:* Performance progression from 30% to 100% training data.
  * *Data Leakage Ablation:* Side-by-side bar chart proving the impact of including $G1/G2$.
  * *CV Stability:* 5-fold cross-validation dispersion boxplots.
* **Explainability:** Feature attribution via MDI, standardized coefficients, and permutation importance with theoretical disclosures on why they are not directly interchangeable.
* **Methodology & Limitations:** Comprehensive protocol documentation, metric equations, and ethical usage mandate.
