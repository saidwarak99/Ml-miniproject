"""
ClassiScope: Performance Analysis of Classification Algorithms on the Student Performance Dataset
Streamlit Interactive Dashboard Application.
Equipped with both rigorous academic metrics and beginner-friendly, student-accessible explanations.
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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header and badge styling */
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

.school-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34D399;
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
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
    color: #CBD5E1;
    font-size: 0.94rem;
    line-height: 1.65;
}

.student-explainer-box {
    background: #132238;
    border: 1px solid #1E3A8A;
    border-left: 4px solid #38BDF8;
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
    color: #E2E8F0;
    font-size: 0.95rem;
    line-height: 1.65;
}

.student-explainer-box h4 {
    color: #38BDF8;
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 1.05rem;
}

.warning-callout {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 4px solid #EF4444;
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    color: #FCA5A5;
    font-size: 0.92rem;
    line-height: 1.6;
    margin: 16px 0;
}

[data-testid="stSidebar"] {
    background-color: #0B132B;
    border-right: 1px solid #1E293B;
}

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
    st.markdown("### 🎓 Quick Concept Guide")
    st.markdown(
        """
        - **What is Machine Learning?**  
          Teaching a computer patterns by showing it past examples.
        - **What are we predicting?**  
          Will a student pass ($G3 \\ge 10$) or need tutoring ($G3 < 10$)?
        - **Why hide early grades?**  
          To see if habits can warn teachers *before* any test is written!
        """
    )
    st.markdown("---")
    st.caption("UCI Student Performance | Portuguese Subject (`student-por.csv`)")


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
    st.markdown('<div class="header-badge">Project Introduction & Student Guide</div>', unsafe_allow_html=True)
    st.title("Performance Analysis of Classification Algorithms")
    st.markdown(
        "An interactive study that tests **6 different computer algorithms** to see how accurately they can predict "
        "whether a student passes or fails school, using only their study habits and lifestyle."
    )

    # Student Friendly Callout Card
    st.markdown(
        """
        <div class="student-explainer-box">
            <h4>💡 Explain Like I'm in School: What is this project all about?</h4>
            Imagine you are a teacher on the first day of high school. You want to know which of your students might 
            struggle later in the year so you can offer them extra tutoring right away—<strong>before they fail any big exam!</strong><br><br>
            Instead of guessing yourself, you give past student survey information (like study hours, school absences, and free time) 
            to <strong>6 different mathematical algorithms</strong> (computer brains). This app tests which algorithm makes the best, 
            fairest guesses, and shows where they make mistakes.
        </div>
        """, unsafe_allow_html=True
    )

    # Key statistics cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Students Studied</div>
                <div class="metric-value">{len(raw_df)}</div>
                <div class="metric-sub">Real records from Portugal</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Habits & Factors Tracked</div>
                <div class="metric-value">{len(num_cols) + len(cat_cols)}</div>
                <div class="metric-sub">{len(num_cols)} numbers • {len(cat_cols)} categories</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        n_pass = int((raw_df["G3"] >= 10).sum())
        p_pass = round((n_pass / len(raw_df)) * 100, 1)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Passed (Grade ≥ 10)</div>
                <div class="metric-value" style="color: #34D399;">{n_pass}</div>
                <div class="metric-sub">{p_pass}% of the class (Majority)</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        n_fail = int((raw_df["G3"] < 10).sum())
        p_fail = round((n_fail / len(raw_df)) * 100, 1)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Needed Help (Grade &lt; 10)</div>
                <div class="metric-value" style="color: #F87171;">{n_fail}</div>
                <div class="metric-sub">{p_fail}% of the class (Minority)</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("### The Core Research Question")
    st.markdown(
        """
        <div class="editorial-note">
            <strong>The Scientific Question:</strong> Which classification algorithm performs best at spotting students who will pass 
            versus fail, and how do their reliability and errors differ when we <strong>strictly hide previous test scores</strong>?
        </div>
        """, unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🎯 How Passing is Defined (The Target)")
        st.markdown(
            """
            In Portuguese secondary schools, student grades are marked from **0 (lowest) to 20 (highest)**.
            - A student who gets **10 or higher** passes the subject.
            - A student who gets **less than 10** fails and must retake it.

            **In our computer models:**
            - **Class 1 (Pass):** Final Grade ($G3$) $\\ge 10$
            - **Class 0 (Fail / Needs Help):** Final Grade ($G3$) $< 10$

            **Why this is tricky for the computer:**
            - **84.6%** of students passed, and only **15.4%** failed!
            - If a lazy computer simply guessed *"Everyone passes!"* for every single student, 
              it would get **84.6% accuracy**, but it would catch **0** failing students! 
              That is why we cannot rely on accuracy alone—we must check **F1-Score** and **Recall**.
            """
        )

    with col_right:
        st.markdown("#### 🚫 The 'No Cheating' Rule (Data Leakage)")
        st.markdown(
            """
            In the real school year, students take three big exams:
            - **$G1$:** Exam in the 1st trimester (autumn)
            - **$G2$:** Exam in the 2nd trimester (winter)
            - **$G3$:** Final grade (summer)

            **Why did we throw away $G1$ and $G2$?**
            - If a computer already knows a student scored an $18/20$ on the winter exam, 
              guessing they will pass in summer is **too easy!** That's called *data leakage* (cheating).
            - Real early-intervention happens in **September**, before any exams exist. 
              So our computer has to rely solely on **habits, attendance, and background**.
            """
        )

    st.markdown("---")
    st.markdown("### 📊 What the Real Student Grades Look Like")
    st.markdown("Below is a histogram showing all 649 final grades. Notice how the dashed yellow line at **10** cuts the class into Passes and Fails:")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=raw_df[raw_df["G3"] < 10]["G3"],
        name="Failed (< 10)",
        marker_color="#F87171",
        nbinsx=10,
        opacity=0.85
    ))
    fig_hist.add_trace(go.Histogram(
        x=raw_df[raw_df["G3"] >= 10]["G3"],
        name="Passed (≥ 10)",
        marker_color="#38BDF8",
        nbinsx=15,
        opacity=0.85
    ))
    fig_hist.add_vline(x=9.5, line_dash="dash", line_color="#FBBF24", annotation_text="Passing Cutoff (10/20)", annotation_position="top left")
    fig_hist.update_layout(
        title="Student Final Grade ($G3$) Distribution",
        barmode="overlay",
        xaxis_title="Final Grade (0 to 20 Scale)",
        yaxis_title="Number of Students",
        **PLOTLY_THEME["layout"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("🔍 Click to see what the raw student data table looks like", expanded=False):
        st.write("Here are the first 10 rows of real student records. Notice columns like `studytime`, `failures`, `absences`, and `internet`:")
        st.dataframe(raw_df.head(10), use_container_width=True)


# =========================================================
# Page 2: Model Benchmark
# =========================================================
elif selected_page == "Model Benchmark":
    st.markdown('<div class="header-badge">The Algorithm Face-Off</div>', unsafe_allow_html=True)
    st.title("Comparing the 6 Machine Learning Algorithms")
    st.markdown(
        "We trained all 6 algorithms on the exact same group of 519 students, "
        "and then tested them on 130 secret test students they had never seen before."
    )

    # Student Friendly Explainer
    with st.expander("🤖 Meet the 6 'Math Robots' (How each algorithm makes its guesses)", expanded=True):
        st.markdown(
            """
            | Algorithm | How it thinks (Simple School Analogy) |
            | :--- | :--- |
            | **1. Logistic Regression** | **The Scorecard:** Assigns plus points or minus points for every habit (e.g., $+2$ for study time, $-3$ for absences). If total score > threshold, it predicts 'Pass'. |
            | **2. K-Nearest Neighbors (KNN)** | **The Study Group ("Birds of a Feather"):** When a new student arrives, it finds the **7 most similar students** from past years. Whatever the majority did, it predicts for the new student. |
            | **3. Decision Tree** | **The Game of 20 Questions:** A step-by-step flowchart: *Did the student fail classes before? $\\rightarrow$ No $\\rightarrow$ Do they study > 5 hours? $\\rightarrow$ Yes $\\rightarrow$ Predict PASS.* |
            | **4. Random Forest** | **The Classroom Committee:** Instead of 1 flowchart, it builds **150 different flowcharts**, lets them all vote, and takes the majority opinion. |
            | **5. Support Vector Machine (SVM)** | **The Fence Builder:** Plots all students on a multi-dimensional map and builds the widest possible boundary fence separating passing students from failing students. |
            | **6. Gradient Boosting** | **The Dedicated Learner:** Builds one flowchart, identifies all the students it got wrong, builds a 2nd flowchart specifically to fix those errors, and repeats 120 times! |
            """
        )

    # Ranking Metric Selector
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        metric_options = [
            "CV F1 (Mean)", "F1-Score", "Macro F1", "Accuracy",
            "Precision", "Recall", "ROC-AUC", "CV Acc (Mean)"
        ]
        selected_metric = st.selectbox(
            "Sort Leaderboard By:",
            metric_options,
            index=0,
            help="Choose which test score determines the winner!"
        )

    sorted_df = summary_df.sort_values(by=selected_metric, ascending=False).reset_index(drop=True)
    top_model = sorted_df.iloc[0]["Model"]
    top_score = sorted_df.iloc[0][selected_metric]

    with ctrl_col2:
        st.markdown(
            f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px 18px; margin-top: 24px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">🏆 TOP PERFORMER FOR {selected_metric.upper()}:</span>
                <strong style="color: #38BDF8; font-size: 1.15rem; margin-left: 8px;">{top_model}</strong>
                <span style="color: #F8FAFC; margin-left: 6px; font-weight: 700;">({top_score:.3f})</span>
            </div>
            """, unsafe_allow_html=True
        )

    # Easy Metric Glossary
    with st.expander("📖 What do all these score column names mean? (Student Cheat-Sheet)", expanded=False):
        st.markdown(
            """
            - **Accuracy:** The total percentage of correct guesses out of all students ($0.80 = 80\\%$).
            - **Precision:** When the model shouts *"This student will PASS!"*, how often was it actually right?
            - **Recall:** Out of all the students who truly passed, how many did the model manage to find?
            - **F1-Score:** The balanced score between Precision and Recall. (Like a decathlon score combining speed and strength).
            - **Macro F1:** An unweighted score giving **equal respect** to both passing and failing students.
            - **ROC-AUC:** How good the model is at separating high performers from struggling students across all strictness levels. ($0.5$ is pure guessing; $1.0$ is perfect).
            - **CV F1 (Mean ± Std):** Cross-validation! We split students into 5 different batches to test if the model gets consistent scores or just got lucky on one test.
            """
        )

    st.markdown("### 🏆 Official Model Leaderboard")
    
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

    st.markdown("---")
    st.markdown("### 📊 Visual Chart Comparisons")

    tab1, tab2 = st.tabs(["Bar Chart: Who Scored Highest?", "Radar Chart: The Full Skills Profile"])

    with tab1:
        fig_bar = px.bar(
            sorted_df,
            x="Model",
            y=selected_metric,
            color="Model",
            color_discrete_map=MODEL_COLORS,
            text_auto=".3f",
            title=f"All 6 Models Ranked by {selected_metric}"
        )
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
            vals.append(vals[0])
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
            title="Radar Profile: Showing Strengths Across All 5 Key Metrics",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# =========================================================
# Page 3: Error Analysis
# =========================================================
elif selected_page == "Error Analysis":
    st.markdown('<div class="header-badge">Investigating Mistakes</div>', unsafe_allow_html=True)
    st.title("Error Analysis: Where Did the Algorithms Get It Wrong?")
    st.markdown(
        "No machine learning model is perfect. In this section, we inspect the exact mistakes "
        "the selected model made on the 130 test students."
    )

    selected_model_name = st.selectbox(
        "Choose an Algorithm to Inspect:",
        list(results_map.keys()),
        index=0
    )
    res = results_map[selected_model_name]

    # Student Friendly Explainer
    st.markdown(
        """
        <div class="student-explainer-box">
            <h4>💡 In School Terms: The 4 Possible Guesses (The Confusion Matrix)</h4>
            Every time the computer predicts a student's fate, only 4 things can happen:<br>
            • <strong>True Positive (TP):</strong> The computer guessed <em>Pass</em>, and the student <strong>Passed</strong>! (Good guess 🎉)<br>
            • <strong>True Negative (TN):</strong> The computer guessed <em>Needs Help</em>, and the student <strong>Needed Help</strong>! (Tutoring offered in time ✅)<br>
            • <strong>False Positive (FP - The Worst Mistake):</strong> The computer guessed <em>Pass</em>, but the student actually <strong>Failed</strong>! (Dangerous: the student fell through the cracks and got no tutoring ❌)<br>
            • <strong>False Negative (FN):</strong> The computer guessed <em>Needs Help</em>, but the student actually <strong>Passed</strong>! (Harmless: student got extra help they didn't strictly need ⚠️)
        </div>
        """, unsafe_allow_html=True
    )

    # 4 Counters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value">{res['test_accuracy']:.1%}</div>
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
                <div class="metric-sub">Passed, but predicted to fail</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">False Positives (FP)</div>
                <div class="metric-value" style="color: #FBBF24;">{res['fp']}</div>
                <div class="metric-sub">Failed, but predicted to pass!</div>
            </div>
            """, unsafe_allow_html=True
        )

    col_cm, col_curves = st.columns([1, 1])

    with col_cm:
        st.markdown("#### The Confusion Matrix Grid")
        cm = res["confusion_matrix"]
        cm_labels = ["Actual Fail (<10)", "Actual Pass (≥10)"]
        pred_labels = ["Pred Fail (<10)", "Pred Pass (≥10)"]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=pred_labels,
            y=cm_labels,
            colorscale=[[0, "#1E293B"], [0.5, "#0284C7"], [1.0, "#38BDF8"]],
            text=[[f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"], [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "#F8FAFC"},
            showscale=False
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix: {selected_model_name}",
            xaxis_title="What the Computer Predicted",
            yaxis_title="What Actually Happened to the Student",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_curves:
        st.markdown("#### Diagnostic Curves")
        sub_tab1, sub_tab2 = st.tabs(["ROC Curve (Separation Power)", "Precision-Recall Curve"])
        
        with sub_tab1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=res["fpr"],
                y=res["tpr"],
                mode="lines",
                line=dict(color="#38BDF8", width=3),
                name=f"Model ROC (AUC = {res['test_roc_auc']:.3f})"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#64748B", dash="dash"),
                name="Pure Coin Toss Line"
            ))
            fig_roc.update_layout(
                title=f"ROC Curve: {selected_model_name}",
                xaxis_title="False Alarm Rate (False Positive Rate)",
                yaxis_title="True Success Rate (True Positive Rate)",
                **PLOTLY_THEME["layout"]
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("💡 The closer the blue curve arches toward the top-left corner, the better the model is!")
            
        with sub_tab2:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=res["pr_recall"],
                y=res["pr_precision"],
                mode="lines",
                line=dict(color="#34D399", width=3),
                name="Precision-Recall"
            ))
            baseline_pr = float(np.mean(res["y_test"]))
            fig_pr.add_hline(y=baseline_pr, line_dash="dash", line_color="#64748B", annotation_text=f"Class Prevalence ({baseline_pr:.2f})")
            fig_pr.update_layout(
                title=f"Precision-Recall Curve: {selected_model_name}",
                xaxis_title="Recall (Coverage of Passing Students)",
                yaxis_title="Precision (Accuracy of 'Pass' Calls)",
                **PLOTLY_THEME["layout"]
            )
            st.plotly_chart(fig_pr, use_container_width=True)


# =========================================================
# Page 4: Experiment Lab
# =========================================================
elif selected_page == "Experiment Lab":
    st.markdown('<div class="header-badge">The Science Laboratory</div>', unsafe_allow_html=True)
    st.title("Experiment Lab & Stress Tests")
    st.markdown(
        "In science, you change one variable at a time to see what happens. "
        "Here are 3 controlled experiments showing how the models behave under stress."
    )

    exp_choice = st.radio(
        "Select an Experiment to Run:",
        [
            "1. Data Diet: Does giving the computer more students make it smarter?",
            "2. The Cheat Test: What happens if we give the computer early exam marks (G1 & G2)?",
            "3. Consistency Check: Does the computer get lucky on certain groups of students?"
        ]
    )

    if "1. Data Diet" in exp_choice:
        st.markdown("### 🧪 Experiment 1: The 'Data Diet' (Learning Curves)")
        st.markdown(
            """
            <div class="student-explainer-box">
                <h4>💡 Simple Explanation: How much homework does a computer need?</h4>
                If a human student only studies 3 practice math problems, they won't do very well on the exam. 
                If they study 50 problems, their score improves!<br>
                Here, we train the computers with only <strong>30% of students</strong>, then <strong>45%</strong>, 
                <strong>60%</strong>, all the way to <strong>100%</strong> to see who learns fastest.
            </div>
            """, unsafe_allow_html=True
        )
        with st.spinner("Computing learning curves across training sizes..."):
            lc_df = get_learning_curve_data()

        fig_lc = px.line(
            lc_df,
            x="Sample Count",
            y="Test F1",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            markers=True,
            title="Test Score Improvement as More Student Records are Provided"
        )
        fig_lc.update_layout(
            xaxis_title="Number of Training Students Given to the Model",
            yaxis_title="Test F1-Score on Unseen Students",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        with st.expander("🔍 Click to view the exact data table"):
            st.dataframe(lc_df, use_container_width=True)

    elif "2. The Cheat Test" in exp_choice:
        st.markdown("### 🧪 Experiment 2: The Data Leakage 'Open Book' Test")
        st.markdown(
            """
            <div class="warning-callout">
                <h4 style="margin-top:0; color:#EF4444;">🚨 The Proof of Why You Cannot Cheat With Prior Grades</h4>
                In school, if you give a student the answers to Quiz 1 and Quiz 2 while they take Quiz 3, their score jumps! 
                That is exactly what happens when ML models include $G1$ and $G2$. 
                Below is the real experiment proving that scores artificially shoot up when you cheat:
            </div>
            """, unsafe_allow_html=True
        )
        with st.spinner("Running leakage comparison..."):
            leak_df = get_leakage_data()

        st.dataframe(
            leak_df.style.format({
                "F1 (Without G1/G2)": "{:.3f}",
                "F1 (With G1/G2 Leakage)": "{:.3f}",
                "F1 Delta (%)": "+{:.2f}%",
                "Acc (Without G1/G2)": "{:.1%}",
                "Acc (With G1/G2 Leakage)": "{:.1%}",
            }).highlight_max(subset=["F1 Delta (%)"], color="#7F1D1D"),
            use_container_width=True
        )

        fig_leak = go.Figure()
        fig_leak.add_trace(go.Bar(
            x=leak_df["Model"],
            y=leak_df["F1 (Without G1/G2)"],
            name="Honest Benchmark (Habits Only, No Prior Grades)",
            marker_color="#38BDF8"
        ))
        fig_leak.add_trace(go.Bar(
            x=leak_df["Model"],
            y=leak_df["F1 (With G1/G2 Leakage)"],
            name="Cheating Benchmark (Includes Prior Trimester Marks)",
            marker_color="#EF4444"
        ))
        fig_leak.update_layout(
            barmode="group",
            title="Comparison: Honest Early Warning vs Leaked Prior Grades",
            yaxis_title="F1-Score",
            yaxis_range=[0.7, 1.0],
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_leak, use_container_width=True)
        st.markdown(
            "Notice how **Decision Tree jumps from 76% accuracy to 90% accuracy**! "
            "That sounds impressive, but it's useless if you want to help students on Day 1 of school."
        )

    elif "3. Consistency Check" in exp_choice:
        st.markdown("### 🧪 Experiment 3: Cross-Validation Consistency")
        st.markdown(
            """
            <div class="student-explainer-box">
                <h4>💡 Simple Explanation: Testing across 5 different student cohorts</h4>
                If a student takes 1 test and gets an A, did they get lucky, or are they genuinely good? 
                To find out, we give them 5 tests! In machine learning, this is called <strong>5-Fold Cross-Validation</strong>. 
                The tighter and smaller the boxplot, the more consistent and trustworthy the algorithm is.
            </div>
            """, unsafe_allow_html=True
        )

        cv_records = []
        for name, r in results_map.items():
            for fold_i, score in enumerate(r["cv_f1_scores"]):
                cv_records.append({
                    "Model": name,
                    "Fold": f"Cohort {fold_i+1}",
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
            title="5-Fold Cross-Validation Score Spread (Consistency Test)"
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
    st.markdown('<div class="header-badge">Peeking Inside the Black Box</div>', unsafe_allow_html=True)
    st.title("Explainability: Which Student Habits Mattered Most?")
    st.markdown(
        "A good scientist doesn't just want a prediction—they want to know **why** the computer made that prediction."
    )

    st.markdown(
        """
        <div class="warning-callout">
            <h4 style="margin-top:0; color:#EF4444;">⚠️ The Golden Rule: Correlation is NOT Causation!</h4>
            Just because the computer noticed that students with higher absences had lower grades 
            <strong>does NOT mean skipping a day automatically makes someone unintelligent</strong>. 
            These are statistical trends in one group of students from 2005. They must NEVER be used to stereotype or judge real people!
        </div>
        """, unsafe_allow_html=True
    )

    c_exp1, c_exp2 = st.columns([1, 1])
    with c_exp1:
        model_for_exp = st.selectbox(
            "Select an Algorithm to Inspect:",
            list(results_map.keys()),
            index=3  # Random Forest default
        )
    with c_exp2:
        top_n = st.slider("Number of Top Factors to Display:", min_value=5, max_value=25, value=12)

    pipe = results_map[model_for_exp]["pipeline"]
    with st.spinner("Extracting feature rankings..."):
        feat_df = extract_feature_importance(
            model_for_exp, pipe, X_train, y_train, X_test, y_test, top_n=top_n
        )

    if not feat_df.empty:
        metric_used = feat_df.iloc[0]["Metric"]
        st.markdown(f"**Math Method Applied:** `{metric_used}`")

        fig_feat = px.bar(
            feat_df.sort_values(by="Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=[MODEL_COLORS[model_for_exp]],
            title=f"Top {top_n} Most Influential Features for {model_for_exp}"
        )
        fig_feat.update_layout(
            xaxis_title="Importance Weight (Higher = Stronger Influence on the Guess)",
            yaxis_title="Student Habit / Attribute",
            **PLOTLY_THEME["layout"]
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with st.expander("📖 What do these feature names mean in plain English?"):
        st.markdown(
            """
            - **`failures`**: Number of past school classes the student previously failed (1, 2, 3, or 4).
            - **`absences`**: Number of days the student was absent from school.
            - **`studytime`**: Weekly study hours (1 = <2 hours, 2 = 2 to 5 hours, 3 = 5 to 10 hours, 4 = >10 hours).
            - **`Medu` / `Fedu`**: Mother's and Father's education level.
            - **`Dalc` / `Walc`**: Workday and Weekend alcohol consumption (self-reported scale 1 to 5).
            - **`higher`**: Whether the student wants to attend university (Yes / No).
            - **`schoolsup`**: Whether the student receives extra educational support from school.
            """
        )

    st.markdown("### Why Different Algorithms Value Different Things")
    st.markdown(
        """
        Notice that if you switch between **Random Forest** and **Logistic Regression**, the ranking of features changes!
        - **Trees (Random Forest)** look for clean splits (e.g., *"Did the student have 0 failures?"*).
        - **Linear Models (Logistic Regression)** look for steady upward or downward ramps.
        - Neither one is "right" or "wrong"—they are just two different ways of looking at the same classroom.
        """
    )


# =========================================================
# Page 6: Methodology & Limitations
# =========================================================
elif selected_page == "Methodology & Limitations":
    st.markdown('<div class="header-badge">Scientific Integrity & Ethics</div>', unsafe_allow_html=True)
    st.title("Methodology, Limitations & Algorithmic Ethics")
    st.markdown(
        "Scientific research requires total honesty about where data came from, "
        "how the experiment was performed, and the serious dangers of misusing AI."
    )

    st.markdown(
        """
        <div class="warning-callout">
            <h3 style="margin-top:0; color:#EF4444;">🛡️ The Ethical Mandate: Protecting Real Students</h3>
            This software was built solely for <strong>educational and scientific benchmark exploration</strong>.<br><br>
            <strong>Under no circumstances should any predictive machine learning model be used to rank, grade, 
            label, or deny opportunities to real students in real schools.</strong><br>
            When computers judge human beings based on parental education or home neighborhood, they risk 
            automating discrimination and treating children as statistics rather than individuals with limitless potential.
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("### 1. Where the Dataset Came From (Dataset Provenance)")
    st.markdown(
        """
        - **Source:** The official **UCI Machine Learning Repository** (Dataset #320).
        - **Students:** 649 high school students from the Alentejo region of Portugal.
        - **Schools:** Gabriel Pereira and Mousinho da Silveira.
        - **Subject:** Portuguese Language class (`student-por.csv`).
        - **Original Research Paper:**  
          > Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*.  
          > Published in Proceedings of 5th Annual Future Business Technology Conference, Porto, Portugal.
        """
    )

    st.markdown("### 2. How the Computer Was Prepared (Preprocessing)")
    st.markdown(
        """
        1. **Standardizing Numbers (`StandardScaler`):**  
           Age is around 15–22, while study time is 1–4, and absences can be 0–32. 
           We put all numbers on a standardized zero-centered scale so algorithms like KNN don't get confused by big numbers.
        2. **Translating Words into Numbers (`OneHotEncoder`):**  
           Computers only understand numbers, not words like `"yes"` or `"no"`. 
           We turn categories into 0s and 1s safely.
        3. **No Cheating Pipeline:**  
           The scaler and encoder were fitted **strictly on the 519 training students**, 
           meaning the computer never got a peek at the 130 test students beforehand.
        """
    )

    st.markdown("### 3. How to Reproduce These Results Yourself")
    st.markdown(
        """
        - **Random Seed:** Locked at `42` so if you run this on any computer in the world, you get the exact same results.
        - **Command to run experiments in terminal:** `python run_experiments.py`
        - **Command to run unit tests:** `python -m unittest tests/test_pipeline.py`
        - **Command to launch this app:** `streamlit run app.py`
        """
    )
