"""
Data loader and dataset preparation module for UCI Student Performance dataset.
Dataset: Portuguese subject (student-por.csv).
Citation: Cortez, P., & Silva, A. M. G. (2008). Using Data Mining to Predict Secondary School Student Performance.
"""

import io
import os
import urllib.request
import zipfile
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "student-por.csv")
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
FALLBACK_GITHUB_URL = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-por.csv"


def ensure_dataset_exists(csv_path: str = CSV_PATH) -> str:
    """
    Ensures that student-por.csv is available locally.
    Downloads from the official UCI repository if not found, with robust error handling.
    """
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Attempt 1: Official UCI ZIP archive
    try:
        req = urllib.request.Request(
            UCI_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ClassiScope/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            outer_zip = zipfile.ZipFile(io.BytesIO(resp.read()))
            # The official UCI archive contains student.zip inside
            if "student.zip" in outer_zip.namelist():
                inner_data = outer_zip.read("student.zip")
                inner_zip = zipfile.ZipFile(io.BytesIO(inner_data))
                inner_zip.extract("student-por.csv", path=os.path.dirname(csv_path))
            elif "student-por.csv" in outer_zip.namelist():
                outer_zip.extract("student-por.csv", path=os.path.dirname(csv_path))
            else:
                raise ValueError("student-por.csv not found in UCI zip files")
        if os.path.exists(csv_path):
            return csv_path
    except Exception as e_uci:
        # Attempt 2: Fallback direct CSV mirror
        try:
            req_fb = urllib.request.Request(
                FALLBACK_GITHUB_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req_fb, timeout=30) as resp_fb:
                content = resp_fb.read()
                # Determine separator
                sample = content[:500].decode("utf-8", errors="ignore")
                sep = ";" if ";" in sample else ","
                df_fb = pd.read_csv(io.BytesIO(content), sep=sep)
                df_fb.to_csv(csv_path, sep=";", index=False)
            return csv_path
        except Exception as e_fb:
            raise RuntimeError(
                f"Failed to fetch dataset from UCI ({e_uci}) and fallback mirror ({e_fb}). "
                f"Please manually place 'student-por.csv' into: {csv_path}"
            )

    return csv_path


def load_raw_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Loads raw student-por data handling semicolon delimiters.
    """
    path = ensure_dataset_exists(csv_path)
    # Autodetect or default to semicolon separator as specified by UCI
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    return df


def prepare_dataset(
    csv_path: str = CSV_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    include_period_grades: bool = False
):
    """
    Prepares features and stratified target for classification.
    
    Target definition:
      - 1 (Higher Grade): G3 >= 10
      - 0 (Lower Grade): G3 < 10
      
    Data Leakage Prevention:
      - By default (include_period_grades=False), G1 and G2 are excluded from features
        because earlier-period grades produce trivial leakage and undermine practical
        early-warning prediction without prior grades.
      - G3 is always excluded from features as it directly defines the target.
    
    Returns:
      X_train, X_test, y_train, y_test, num_cols, cat_cols, raw_df
    """
    df = load_raw_data(csv_path)
    
    # Define binary target
    target = (df["G3"] >= 10).astype(int)
    
    # Exclude G3 always
    cols_to_drop = ["G3"]
    if not include_period_grades:
        cols_to_drop.extend(["G1", "G2"])
        
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = target
    
    # Separate numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Stratified split to preserve class imbalance in both train and test partitions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, num_cols, cat_cols, df
