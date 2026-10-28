"""
Model architecture definitions and pipeline construction for ClassiScope.
Implements the six required algorithms wrapped in scikit-learn Pipelines:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVC with probability=True)
6. Gradient Boosting
"""

from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def create_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Creates a ColumnTransformer that scales numeric features and
    one-hot encodes categorical features.
    Configured to handle unseen categories safely without throwing errors.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_model_registry(random_state: int = 42) -> Dict[str, dict]:
    """
    Returns metadata and raw estimators for all six classification algorithms.
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=random_state,
                solver="lbfgs"
            ),
            "family": "Linear",
            "supports_feature_importance": True,
            "importance_type": "coefficients",
            "description": "Standard linear decision boundary with L2 regularization."
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(
                n_neighbors=7,
                weights="uniform",
                metric="minkowski",
                p=2
            ),
            "family": "Instance-based",
            "supports_feature_importance": False,
            "importance_type": "permutation",
            "description": "Non-parametric distance-based classification in normalized feature space."
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=random_state
            ),
            "family": "Tree-based",
            "supports_feature_importance": True,
            "importance_type": "gini_importance",
            "description": "Single orthogonal decision tree with controlled depth to mitigate overfitting."
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=6,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1
            ),
            "family": "Ensemble (Bagging)",
            "supports_feature_importance": True,
            "importance_type": "mdi",
            "description": "Bootstrap ensemble of de-correlated decision trees with feature sub-sampling."
        },
        "Support Vector Machine": {
            "estimator": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=random_state
            ),
            "family": "Kernel-based",
            "supports_feature_importance": False,
            "importance_type": "permutation",
            "description": "Radial Basis Function (RBF) margin maximizer in high-dimensional space."
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=3,
                min_samples_split=6,
                min_samples_leaf=3,
                random_state=random_state
            ),
            "family": "Ensemble (Boosting)",
            "supports_feature_importance": True,
            "importance_type": "deviance_reduction",
            "description": "Sequential additive boosting optimizing binary log-loss."
        },
    }


def build_pipeline(
    model_name: str,
    num_cols: List[str],
    cat_cols: List[str],
    random_state: int = 42
) -> Pipeline:
    """
    Constructs a complete scikit-learn Pipeline with preprocessor and classifier.
    Ensures no data leakage by keeping preprocessor fitted strictly alongside classifier.
    """
    registry = get_model_registry(random_state)
    if model_name not in registry:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(registry.keys())}")
        
    estimator = registry[model_name]["estimator"]
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    return pipeline


def build_all_pipelines(
    num_cols: List[str],
    cat_cols: List[str],
    random_state: int = 42
) -> Dict[str, Pipeline]:
    """
    Builds pipelines for all six algorithms.
    """
    registry = get_model_registry(random_state)
    return {
        name: build_pipeline(name, num_cols, cat_cols, random_state)
        for name in registry.keys()
    }
