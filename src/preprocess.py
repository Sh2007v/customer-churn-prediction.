"""Data loading, cleaning, encoding and scaling.

Cleaning (clean_data) is rule-based and safe to run on the full dataset.
Imputing, scaling and encoding live inside a scikit-learn ColumnTransformer
so they are FIT ON TRAINING DATA ONLY (no data leakage) when the transformer
is part of a Pipeline that is fitted on the training split.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_data.csv"

ID_COLUMN = "customer_id"
TARGET = "churn"
NUMERIC = ["tenure_months", "add_on_services", "monthly_charges",
           "monthly_usage_gb", "logins_per_month", "support_calls"]
CATEGORICAL = ["contract", "payment_method", "internet_plan"]
FEATURES = NUMERIC + CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it first: python src/generate_data.py")
    return pd.read_csv(path)


def clean_data(df):
    """Remove duplicates, impossible values and rows without a valid label.

    Returns (clean_df, report). The id column is kept so predictions can be
    traced back to a customer, but it is never used as a model feature.
    """
    report = {"rows_in": len(df)}
    df = df.copy()

    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    report["missing_values_found"] = {
        c: int(n) for c, n in df[FEATURES].isna().sum().items() if n > 0}

    # Target: Yes/No -> 1/0
    label = df[TARGET].astype(str).str.strip().str.title()
    df[TARGET] = label.map({"Yes": 1, "No": 0})
    report["rows_with_invalid_label_removed"] = int(df[TARGET].isna().sum())
    df = df[df[TARGET].notna()]
    df[TARGET] = df[TARGET].astype(int)

    # Impossible values (negative counts/charges). NaN is handled later.
    impossible = (df[NUMERIC] < 0).any(axis=1)
    report["impossible_rows_removed"] = int(impossible.sum())
    df = df[~impossible]

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def build_preprocessor():
    """Impute -> scale numbers; impute -> one-hot encode categories."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", categorical, CATEGORICAL)])


def split_data(df):
    """80/20 stratified split (same churn ratio in both parts, repeatable).

    X keeps only the model features (no id, no target); the original index is
    preserved so test rows can be traced back to customer_id.
    """
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)


if __name__ == "__main__":
    raw = load_data()
    clean, rep = clean_data(raw)
    print("Cleaning report:", rep)
    X_tr, X_te, y_tr, y_te = split_data(clean)
    print(f"Train: {len(X_tr)} rows ({y_tr.mean():.1%} churn) | "
          f"Test: {len(X_te)} rows ({y_te.mean():.1%} churn)")
