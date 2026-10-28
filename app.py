"""
ClassiScope: Performance Analysis of Classification Algorithms on the Student Performance Dataset
Streamlit Interactive Dashboard Application.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Append project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import prepare_dataset, ensure_dataset_exists, load_raw_data
from src.evaluation import (
    evaluate_all_models,
    extract_feature_importance,
    run_training_size_experiment,
    run_leakage_comparison_experiment,
)
from src.models import get_model_registry

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ClassiScope | ML Classification Benchmark",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Editorial Dark CSS Styling
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Base typography and background overrides */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header and card styling */
.header-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38BDF8;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.metric-card:hover {
    border-color: #38BDF8;
}

.metric-title {
    color: #94A3B8;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.2;
}

.metric-sub {
    color: #38BDF8;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 4px;
}

.editorial-note {
    background: #111827;
    border-left: 3px solid #38BDF8;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
    color: #CBD5E1;
    font-size: 0.92rem;
    line-height: 1.6;
}

.warning-callout {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 4px solid #EF4444;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    color: #FCA5A5;
    font-size: 0.9rem;
    margin: 16px 0;
}

/* Sidebar refined styling */
[data-testid="stSidebar"] {
    background-color: #0B132B;
    border-right: 1px solid #1E293B;
}

/* Dataframe polish */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Plotly Dark Theme Palette Helpers
# ---------------------------------------------------------
PLOTLY_THEME = {
    "layout": {
        "paper_bgcolor": "#1E293B",
        "plot_bgcolor": "#0F172A",
        "font": {"color": "#E2E8F0", "family": "Plus Jakarta Sans, sans-serif"},
        "xaxis": {
            "gridcolor": "#334155",
            "linecolor": "#475569",
            "zerolinecolor": "#334155"
        },
        "yaxis": {
            "gridcolor": "#334155",
            "linecolor": "#475569",
            "zerolinecolor": "#334155"
        },
        "margin": {"t": 40, "b": 40, "l": 50, "r": 30},
    }
}

MODEL_COLORS = {
    "Logistic Regression": "#38BDF8",
    "K-Nearest Neighbors": "#818CF8",
    "Decision Tree": "#F472B6",
    "Random Forest": "#34D399",
    "Support Vector Machine": "#FBBF24",
    "Gradient Boosting": "#FB923C",
}


# ---------------------------------------------------------
# Data & Experiment Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset():
    ensure_dataset_exists()
    return prepare_dataset(include_period_grades=False)


@st.cache_data(show_spinner=False)
def get_benchmark_results():
    X_train, X_test, y_train, y_test, num_cols, cat_cols, raw_df = get_dataset()
    summary_df, results_map = evaluate_all_models(
        X_train, y_train, X_test, y_test, num_cols, cat_cols, cv_folds=5, random_state=42
    )
    return summary_df, results_map, raw_df, X_train, X_test, y_train, y_test, num_cols, cat_cols


@st.cache_data(show_spinner=False)
def get_learning_curve_data():
    X_train, X_test, y_train, y_test, num_cols, cat_cols, _ = get_dataset()
    return run_training_size_experiment(X_train, y_train, X_test, y_test, num_cols, cat_cols)


@st.cache_data(show_spinner=False)
def get_leakage_data():
    raw_df = load_raw_data()
    return run_leakage_comparison_experiment(raw_df)


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="header-badge">ClassiScope v1.0</div>', unsafe_allow_html=True)
    st.markdown("## Navigation")
    
    pages = [
        "Overview",
        "Model Benchmark",
        "Error Analysis",
        "Experiment Lab",
        "Explainability",
        "Methodology & Limitations"
    ]
    
    selected_page = st.radio(
        "Select Section",
        pages,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Experiment Parameters")
    st.markdown(
        """
        - **Target:** $G3 \\ge 10$ (Binary)
        - **Data Split:** Stratified 80/20
        - **Cross-Validation:** 5-Fold Stratified
        - **Seed:** Fixed (`42`)
        - **Leakage Control:** $G1, G2$ Excluded
        """
    )
    st.markdown("---")
    st.caption("UCI Student Performance | Portuguese (`student-por.csv`)")


# ---------------------------------------------------------
# Load Cached Data
# ---------------------------------------------------------
try:
    summary_df, results_map, raw_df, X_train, X_test, y_train, y_test, num_cols, cat_cols = get_benchmark_results()
except Exception as e:
    st.error(f"Error loading benchmark experiment data: {e}")
    st.stop()


# =========================================================
# Page 1: Overview
# =========================================================
if selected_page == "Overview":
    st.markdown('<div class="header-badge">Project Introduction & Foundation</div>', unsafe_allow_html=True)
    st.title("Performance Analysis of Classification Algorithms")
    st.markdown(
        "An empirical benchmarking platform comparing six supervised classification paradigms "
        "on the UCI Student Performance dataset under rigorous leakage-free validation."
    )

    # Key statistics cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{len(raw_df)}</div>
                <div class="metric-sub">Portuguese Subject</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Features Evaluated</div>
                <div class="metric-value">{len(num_cols) + len(cat_cols)}</div>
                <div class="metric-sub">{len(num_cols)} Numeric • {len(cat_cols)} Categorical</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        n_pass = int((raw_df["G3"] >= 10).sum())
        p_pass = round((n_pass / len(raw_df)) * 100, 1)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Higher Grade (≥ 10)</div>
                <div class="metric-value">{n_pass}</div>
                <div class="metric-sub">{p_pass}% Class 1 (Majority)</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        n_fail = int((raw_df["G3"] < 10).sum())
        p_fail = round((n_fail / len(raw_df)) * 100, 1)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Lower Grade (&lt; 10)</div>
                <div class="metric-value">{n_fail}</div>
                <div class="metric-sub">{p_fail}% Class 0 (Imbalanced)</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("### Research Question & Hypothesis")
    st.markdown(
        """
        <div class="editorial-note">
            <strong>Core Research Question:</strong> How do diverse classification algorithms 
            (linear boundaries, distance neighborhoods, recursive partitioning, bootstrap ensembles, 
            support vector kernels, and sequential boosting) compare in predicting whether a student achieves a passing final grade, 
            and how do their reliability and error patterns differ when prior-period evaluations are deliberately withheld?
        </div>
        """, unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Target Definition & Discretization")
        st.markdown(
            """
            The raw dataset records $G3$ as a continuous final score ranging from $0$ to $20$ points. 
            In the Portuguese educational system, a minimum grade of **10/20** is required for promotion:
            - **Class 1 (Higher Grade / Pass):** $G3 \\ge 10$
            - **Class 0 (Lower Grade / Fail):** $G3 < 10$

            **Academic Limitations of this Threshold:**
            1. **Information Loss:** Discretizing continuous performance into a hard threshold discards 
               distinctions between high-achieving ($18/20$) and marginally passing ($10/20$) students.
            2. **Boundary Sensitivity:** A student scoring $9.4$ is placed in Class 0, while $10.0$ is Class 1, 
               introducing classification volatility around borderline scores.
            3. **Severe Class Imbalance:** **84.6%** of students pass, which implies a naive classifier 
               predicting 'Pass' for everyone achieves $84.6\\%$ accuracy while having **zero sensitivity** to at-risk students.
            """
        )

    with col_right:
        st.markdown("#### Primary Experiment Leakage Control")
        st.markdown(
            """
            In many historical machine learning exercises, models include $G1$ (first period grade) 
            and $G2$ (second period grade). In this benchmark:
            - **$G1$ and $G2$ are strictly removed** from the feature matrix.
            - **Rationale:** Using prior-period grades turns the problem into a trivial autoregressive prediction ($r > 0.8$), 
              causing massive data leakage and making the model useless for semester-start academic support.
            - **$G3$ is naturally removed**, as it is the target source.
            """
        )

    st.markdown("---")
    st.markdown("### Raw Target Distribution and Class Disparity")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=raw_df[raw_df["G3"] < 10]["G3"],
        name="Lower Grade (< 10)",
        marker_color="#F87171",
        nbinsx=10,
        opacity=0.85
    ))
    fig_hist.add_trace(go.Histogram(
        x=raw_df[raw_df["G3"] >= 10]["G3"],
        name="Higher Grade (≥ 10)",
        marker_color="#38BDF8",
        nbinsx=15,
        opacity=0.85
    ))
    fig_hist.add_vline(x=9.5, line_dash="dash", line_color="#FBBF24", annotation_text="Threshold (10)", annotation_position="top left")
    fig_hist.update_layout(
        title="Continuous G3 Distribution with Binary Cutoff",
        barmode="overlay",
        xaxis_title="Final Grade (G3: 0 - 20 Scale)",
        yaxis_title="Student Count",
        **PLOTLY_THEME["layout"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Dataset preview
    with st.expander("Explore Raw Dataset Sample (First 10 Rows)", expanded=False):
        st.dataframe(raw_df.head(10), use_container_width=True)


# =========================================================
# Page 2: Model Benchmark
# =========================================================
elif selected_page == "Model Benchmark":
    st.markdown('<div class="header-badge">Comparative Performance Benchmark</div>', unsafe_allow_html=True)
    st.title("Algorithm Comparison Across Evaluation Metrics")
    st.markdown(
        "All six models were trained on identical stratified partitions ($N_{train} = 519$, $N_{test} = 130$) "
        "with isolated preprocessing and 5-fold cross-validation."
    )

    # Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        metric_options = [
            "CV F1 (Mean)", "F1-Score", "Macro F1", "Accuracy",
            "Precision", "Recall", "ROC-AUC", "CV Acc (Mean)"
        ]
        selected_metric = st.selectbox(
            "Primary Ranking Metric",
            metric_options,
            index=0,
            help="Select the metric used to sort the leaderboard and plot comparisons."
        )

    sorted_df = summary_df.sort_values(by=selected_metric, ascending=False).reset_index(drop=True)
    top_model = sorted_df.iloc[0]["Model"]
    top_score = sorted_df.iloc[0][selected_metric]

    with ctrl_col2:
        st.markdown(
            f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px 18px; margin-top: 24px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">LEADER FOR {selected_metric.upper()}:</span>
                <strong style="color: #38BDF8; font-size: 1.15rem; margin-left: 8px;">{top_model}</strong>
                <span style="color: #F8FAFC; margin-left: 6px; font-weight: 700;">({top_score:.4f})</span>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("### Benchmark Leaderboard")
    
    # Format display table nicely
    display_df = sorted_df.copy()
    format_dict = {
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}",
        "Macro F1": "{:.3f}",
        "ROC-AUC": "{:.3f}",
        "CV F1 (Mean)": "{:.3f}",
        "CV F1 (Std)": "±{:.3f}",
        "CV Acc (Mean)": "{:.3f}",
        "CV Acc (Std)": "±{:.3f}",
    }
    st.dataframe(
        display_df.style.format(format_dict).highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "Macro F1", "ROC-AUC", "CV F1 (Mean)", "CV Acc (Mean)"],
            color="#0E4429"
        ),
        use_container_width=True
    )

    st.markdown(
        """
        <p style="font-size: 0.85rem; color: #94A3B8;">
            * CV metrics denote 5-fold Stratified Cross-Validation on the training set (mean ± standard deviation). 
            Holdout test metrics reflect unseen test generalization. Models are strictly not tuned against the test partition.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Interactive Metric Comparisons")

    tab1, tab2 = st.tabs(["Selected Metric Bar Comparison", "Multi-Metric Radar Profiling"])

    with tab1:
        fig_bar = px.bar(
            sorted_df,
            x="Model",
            y=selected_metric,
            color="Model",
            color_discrete_map=MODEL_COLORS,
            text_auto=".3f",
            title=f"Models Ranked by {selected_metric}"
        )
        # Add error bars if plotting CV metric
        if selected_metric == "CV F1 (Mean)":
            fig_bar.update_traces(error_y=dict(type="data", array=sorted_df["CV F1 (Std)"]))
        elif selected_metric == "CV Acc (Mean)":
            fig_bar.update_traces(error_y=dict(type="data", array=sorted_df["CV Acc (Std)"]))

        fig_bar.update_layout(
            showlegend=False,
            yaxis_range=[0.5, 1.0],
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        radar_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        fig_radar = go.Figure()
        for idx, row in sorted_df.iterrows():
            m_name = row["Model"]
            vals = [row[m] for m in radar_metrics]
            vals.append(vals[0])  # close polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                fillcolor=f"rgba({int(MODEL_COLORS[m_name][1:3], 16)}, {int(MODEL_COLORS[m_name][3:5], 16)}, {int(MODEL_COLORS[m_name][5:7], 16)}, 0.15)",
                line=dict(color=MODEL_COLORS[m_name], width=2),
                name=m_name
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.4, 1.0], gridcolor="#334155"),
                angularaxis=dict(gridcolor="#334155")
            ),
            title="Holistic Multi-Metric Radar Comparison",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# =========================================================
# Page 3: Error Analysis
# =========================================================
elif selected_page == "Error Analysis":
    st.markdown('<div class="header-badge">Diagnostic Inspection</div>', unsafe_allow_html=True)
    st.title("Error Analysis & Diagnostic Curves")
    st.markdown(
        "Examine model prediction distributions, confusion matrix dynamics, "
        "and false-positive versus false-negative rates on the holdout test set ($N = 130$)."
    )

    selected_model_name = st.selectbox(
        "Select Model to Inspect",
        list(results_map.keys()),
        index=0
    )
    res = results_map[selected_model_name]

    # Metrics top cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value">{res['test_accuracy']:.3f}</div>
                <div class="metric-sub">{int(res['test_accuracy']*len(res['y_test']))} / {len(res['y_test'])} correct</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">True Positives (TP)</div>
                <div class="metric-value" style="color: #34D399;">{res['tp']}</div>
                <div class="metric-sub">Correctly Predicted Pass</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">False Negatives (FN)</div>
                <div class="metric-value" style="color: #F87171;">{res['fn']}</div>
                <div class="metric-sub">Passed, but Predicted Fail</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">False Positives (FP)</div>
                <div class="metric-value" style="color: #FBBF24;">{res['fp']}</div>
                <div class="metric-sub">Failed, but Predicted Pass</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="editorial-note">
            <strong>Practical Real-World Interpretation of Errors:</strong><br>
            • <strong>False Positives (FP):</strong> An at-risk student (actual score < 10) is predicted to pass. 
            In academic intervention systems, this error is the most critical because the student is denied early tutoring.<br>
            • <strong>False Negatives (FN):</strong> A passing student (actual score ≥ 10) is flagged as at-risk. 
            While less harmful pedagogically, high false negatives misallocate limited academic counseling resources.
        </div>
        """, unsafe_allow_html=True
    )

    col_cm, col_curves = st.columns([1, 1])

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = res["confusion_matrix"]
        cm_labels = ["Actual Lower (<10)", "Actual Higher (≥10)"]
        pred_labels = ["Pred Lower (<10)", "Pred Higher (≥10)"]
        
        # Interactive heatmap
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=pred_labels,
            y=cm_labels,
            colorscale=[[0, "#1E293B"], [0.5, "#0284C7"], [1.0, "#38BDF8"]],
            text=[[f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"], [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "#F8FAFC"},
            showscale=False
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix: {selected_model_name}",
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_curves:
        st.markdown("#### ROC & Precision-Recall Curves")
        sub_tab1, sub_tab2 = st.tabs(["ROC Curve", "Precision-Recall Curve"])
        
        with sub_tab1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=res["fpr"],
                y=res["tpr"],
                mode="lines",
                line=dict(color="#38BDF8", width=3),
                name=f"ROC (AUC = {res['test_roc_auc']:.3f})"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#64748B", dash="dash"),
                name="Chance Baseline"
            ))
            fig_roc.update_layout(
                title=f"ROC Curve: {selected_model_name}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                **PLOTLY_THEME["layout"]
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with sub_tab2:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=res["pr_recall"],
                y=res["pr_precision"],
                mode="lines",
                line=dict(color="#34D399", width=3),
                name="Precision-Recall"
            ))
            # Baseline PR is positive class prevalence
            baseline_pr = float(np.mean(res["y_test"]))
            fig_pr.add_hline(y=baseline_pr, line_dash="dash", line_color="#64748B", annotation_text=f"Prevalence ({baseline_pr:.2f})")
            fig_pr.update_layout(
                title=f"Precision-Recall Curve: {selected_model_name}",
                xaxis_title="Recall",
                yaxis_title="Precision",
                **PLOTLY_THEME["layout"]
            )
            st.plotly_chart(fig_pr, use_container_width=True)


# =========================================================
# Page 4: Experiment Lab
# =========================================================
elif selected_page == "Experiment Lab":
    st.markdown('<div class="header-badge">Controlled Empirical Experiments</div>', unsafe_allow_html=True)
    st.title("Experiment Lab & Diagnostic Stress-Tests")
    st.markdown(
        "Execute controlled variations to analyze sample efficiency, empirical data leakage, "
        "and fold-to-fold cross-validation stability."
    )

    exp_choice = st.radio(
        "Select Experiment",
        [
            "1. Training Data Proportion (Learning Curve)",
            "2. Data Leakage Ablation (With vs Without G1/G2)",
            "3. Cross-Validation Stability Analysis"
        ],
        horizontal=True
    )

    if "1. Training Data" in exp_choice:
        st.markdown("### Sample Efficiency: Performance Across Training Set Proportions")
        st.markdown(
            """
            This experiment trains all six algorithms on incremental subsets of the training data 
            (from 30% up to 100% of training data) while keeping the holdout test set fixed. 
            It evaluates which algorithms reach asymptotic performance fastest.
            """
        )
        with st.spinner("Computing learning curves..."):
            lc_df = get_learning_curve_data()

        fig_lc = px.line(
            lc_df,
            x="Sample Count",
            y="Test F1",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            markers=True,
            title="Test F1-Score Progression Across Training Sample Sizes"
        )
        fig_lc.update_layout(
            xaxis_title="Number of Training Instances",
            yaxis_title="Test F1-Score",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        with st.expander("View Training Proportion Data Table"):
            st.dataframe(lc_df, use_container_width=True)

    elif "2. Data Leakage" in exp_choice:
        st.markdown("### Data Leakage Ablation Study")
        st.markdown(
            """
            <div class="warning-callout">
                <strong>Crucial Methodological Demonstration:</strong> 
                Many student success models report artificially inflated F1 and accuracy metrics (95%+) 
                because they naively include intermediate trimester marks ($G1, G2$). 
                Below is the empirical proof of this leakage effect.
            </div>
            """, unsafe_allow_html=True
        )
        with st.spinner("Executing leakage comparison..."):
            leak_df = get_leakage_data()

        st.dataframe(
            leak_df.style.format({
                "F1 (Without G1/G2)": "{:.3f}",
                "F1 (With G1/G2 Leakage)": "{:.3f}",
                "F1 Delta (%)": "+{:.2f}%",
                "Acc (Without G1/G2)": "{:.3f}",
                "Acc (With G1/G2 Leakage)": "{:.3f}",
            }).highlight_max(subset=["F1 Delta (%)"], color="#7F1D1D"),
            use_container_width=True
        )

        fig_leak = go.Figure()
        fig_leak.add_trace(go.Bar(
            x=leak_df["Model"],
            y=leak_df["F1 (Without G1/G2)"],
            name="Realistic Benchmark (No Prior Grades)",
            marker_color="#38BDF8"
        ))
        fig_leak.add_trace(go.Bar(
            x=leak_df["Model"],
            y=leak_df["F1 (With G1/G2 Leakage)"],
            name="Leaked Pipeline (Includes G1 & G2)",
            marker_color="#EF4444"
        ))
        fig_leak.update_layout(
            barmode="group",
            title="Impact of G1 and G2 Leakage on F1-Score Across Models",
            yaxis_title="F1-Score",
            yaxis_range=[0.7, 1.0],
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_leak, use_container_width=True)

    elif "3. Cross-Validation" in exp_choice:
        st.markdown("### Fold-to-Fold Cross-Validation Stability")
        st.markdown(
            "Examines the variance and dispersion across all 5 stratified cross-validation folds on the training set. "
            "Lower variance indicates higher algorithmic stability when exposed to varying student cohorts."
        )

        cv_records = []
        for name, r in results_map.items():
            for fold_i, score in enumerate(r["cv_f1_scores"]):
                cv_records.append({
                    "Model": name,
                    "Fold": f"Fold {fold_i+1}",
                    "F1-Score": score
                })
        cv_plot_df = pd.DataFrame(cv_records)

        fig_box = px.box(
            cv_plot_df,
            x="Model",
            y="F1-Score",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            points="all",
            title="5-Fold Stratified CV F1-Score Dispersion"
        )
        fig_box.update_layout(
            showlegend=False,
            yaxis_range=[0.85, 0.98],
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_box, use_container_width=True)


# =========================================================
# Page 5: Explainability
# =========================================================
elif selected_page == "Explainability":
    st.markdown('<div class="header-badge">Model Interpretability & Feature Attribution</div>', unsafe_allow_html=True)
    st.title("Feature Attribution & Explainability")
    st.markdown(
        "Inspect the empirical feature contributions of each classification paradigm."
    )

    st.markdown(
        """
        <div class="warning-callout">
            <strong>Ethical & Scientific Advisory:</strong> 
            Feature importances reflect statistical correlations within this sample from two Portuguese schools (2005–2006). 
            <strong>They must NEVER be interpreted as causal claims</strong> about student background, intelligence, 
            family structure, or societal stereotypes.
        </div>
        """, unsafe_allow_html=True
    )

    c_exp1, c_exp2 = st.columns([1, 1])
    with c_exp1:
        model_for_exp = st.selectbox(
            "Select Model for Feature Attribution",
            list(results_map.keys()),
            index=3  # Default Random Forest
        )
    with c_exp2:
        top_n = st.slider("Number of Top Features", min_value=5, max_value=25, value=12)

    pipe = results_map[model_for_exp]["pipeline"]
    with st.spinner("Extracting feature attributions..."):
        feat_df = extract_feature_importance(
            model_for_exp, pipe, X_train, y_train, X_test, y_test, top_n=top_n
        )

    if not feat_df.empty:
        metric_used = feat_df.iloc[0]["Metric"]
        st.markdown(f"**Metric Applied:** `{metric_used}`")

        fig_feat = px.bar(
            feat_df.sort_values(by="Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=[MODEL_COLORS[model_for_exp]],
            title=f"Top {top_n} Features: {model_for_exp}"
        )
        fig_feat.update_layout(
            xaxis_title="Attribution Score",
            yaxis_title="Feature Name",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("### Why Feature Importance Metrics are Not Interchangeable")
    st.markdown(
        """
        Different classification families quantify feature contributions through fundamentally distinct mathematical lenses:
        
        | Paradigm | Method | Mathematical Meaning | Limitations |
        | :--- | :--- | :--- | :--- |
        | **Tree / Ensembles** | Mean Decrease Impurity (MDI) | Total Gini purity gain aggregated across splits | Heavily biased toward high-cardinality and continuous variables. |
        | **Logistic Regression** | Standardized $\\beta$ Coefficients | Change in log-odds per standard deviation shift | Assumes strictly linear and monotonic relationships. |
        | **KNN / SVM** | Permutation Importance | Drop in holdout F1 when feature values are permuted | Dependent on test set size; sensitive to correlated feature collinearity. |
        """
    )


# =========================================================
# Page 6: Methodology & Limitations
# =========================================================
elif selected_page == "Methodology & Limitations":
    st.markdown('<div class="header-badge">Academic Protocol & Governance</div>', unsafe_allow_html=True)
    st.title("Methodology, Limitations & Ethics")
    st.markdown(
        "Complete transparency regarding dataset provenance, experimental design, metric definitions, and algorithmic ethics."
    )

    st.markdown(
        """
        <div class="warning-callout">
            <h4 style="margin-top:0; color:#EF4444;">ETHICAL AND RESPONSIBLE USE MANDATE</h4>
            This software and all benchmark models are designed <strong>strictly for educational, scientific, 
            and algorithmic benchmarking purposes</strong>.<br><br>
            <strong>These models MUST NOT be used to label, rank, sort, deny opportunities to, or make 
            consequential academic decisions about real students.</strong> Automated decision systems built on demographic 
            or behavioral markers risk perpetuating systemic biases and penalizing vulnerable individuals.
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("### 1. Dataset Provenance & Citation")
    st.markdown(
        """
        - **Source:** UCI Machine Learning Repository (Dataset ID: 320)
        - **Subjects:** Secondary school students in the Alentejo region of Portugal (schools: Gabriel Pereira and Mousinho da Silveira).
        - **Dataset file used:** `student-por.csv` (Portuguese Language subject, $N = 649$, 33 original attributes).
        - **Original Citation:**  
          > Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*.  
          > In Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC 2008), Porto, Portugal, pp. 5–12.
        """
    )

    st.markdown("### 2. Preprocessing & Leakage Elimination")
    st.markdown(
        """
        1. **Feature Separation:**
           - **Numeric features (13 attributes):** `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`.
           - **Categorical features (17 attributes):** `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.
        2. **Pipeline Construction:**
           - `StandardScaler()` applied to numeric features.
           - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to categorical features.
           - All transformers were bundled inside a scikit-learn `Pipeline` alongside the estimator to guarantee that transformations were **strictly fit on training partitions only**.
        3. **Data Leakage Mitigation:**
           - Intermediate trimester marks ($G1$ and $G2$) and final score ($G3$) were removed prior to feature processing.
        """
    )

    st.markdown("### 3. Evaluation Metric Definitions")
    st.markdown(
        """
        - **Accuracy:** Proportion of overall correct classifications: $\\frac{TP + TN}{TP + TN + FP + FN}$.
        - **Precision (Positive Class):** $\\frac{TP}{TP + FP}$. Measures how many students predicted to pass actually passed.
        - **Recall (Positive Class):** $\\frac{TP}{TP + FN}$. Measures the proportion of passing students correctly recognized.
        - **F1-Score:** Harmonic mean of precision and recall: $2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall}$.
        - **Macro F1:** Unweighted average of F1 scores across both Class 0 and Class 1, highlighting minority class performance.
        - **ROC-AUC:** Area under the Receiver Operating Characteristic curve, measuring separation ability across all decision thresholds.
        - **5-Fold Stratified Cross-Validation:** The training set ($N=519$) is divided into 5 stratified folds. The model is trained on 4 folds and tested on the 5th, repeated 5 times to compute mean and standard deviation.
        """
    )

    st.markdown("### 4. Reproducibility Configuration")
    st.markdown(
        """
        - **Random Seed:** Fixed at `42` across dataset partitioning, k-fold splits, and stochastic estimators.
        - **Python Version:** 3.12
        - **Core Libraries:** `scikit-learn>=1.5.0`, `pandas>=2.2.0`, `plotly>=6.0.0`, `streamlit>=1.40.0`.
        - **Repository Structure:**
          - `app.py`: Streamlit dashboard
          - `src/data_loader.py`: Download & preprocessing pipeline
          - `src/models.py`: Pipeline definitions & estimator registry
          - `src/evaluation.py`: Metric computation, CV, error analysis, experiments
          - `src/report_generator.py`: Automated report generator
          - `run_experiments.py`: Headless CLI evaluation script
          - `report/results_summary.md`: Auto-generated academic findings report
        """
    )
