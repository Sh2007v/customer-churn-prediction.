"""Predict churn probability for new customers (with input validation).

Usage (from the project root):
    python src/predict.py                       # predict 4 demo customers
    python src/predict.py --input new.csv       # predict customers in a CSV
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocess import CATEGORICAL, FEATURES, NUMERIC, ROOT

MODEL_PATH = ROOT / "models" / "churn_model.pkl"

LIMITATIONS = ("These are estimates based on sample data and past patterns. "
               "They are not certain, and some predictions will be wrong.")

# probability bands used for the risk label
LOW_BELOW, HIGH_FROM = 0.30, 0.60


def risk_label(p):
    if p >= HIGH_FROM:
        return "High"
    return "Medium" if p >= LOW_BELOW else "Low"


def load_bundle(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file {path} not found. Train it first: "
            "python src/train_model.py")
    return joblib.load(path)


def validate_input(df, bundle):
    """Return (clean numeric/categorical frame, warnings) or raise ValueError
    with a clear message. Nothing is predicted for invalid input."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("No customer data was provided.")
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required field(s): {', '.join(missing_cols)}")

    X = df[FEATURES].copy()
    problems, warnings = [], []

    for col in NUMERIC:
        values = pd.to_numeric(X[col], errors="coerce")
        bad = X.index[values.isna()].tolist()
        if bad:
            problems.append(f"{col}: missing or non-numeric value in row(s) {bad}")
        neg = X.index[values < 0].tolist()
        if neg:
            problems.append(f"{col}: negative value not allowed in row(s) {neg}")
        X[col] = values
        lo, hi = bundle["numeric_ranges"][col]
        outside = X.index[(values < lo) | (values > hi)].tolist()
        if outside and not neg:
            warnings.append(
                f"{col} is outside the training range ({lo:g} to {hi:g}) in "
                f"row(s) {outside}; the estimate is less reliable there.")

    for col in CATEGORICAL:
        s = X[col]
        empty = X.index[s.isna() | (s.astype(str).str.strip() == "")].tolist()
        if empty:
            problems.append(f"{col}: missing value in row(s) {empty}")
        known = bundle["categories"][col]
        unknown = X.index[~s.isin(known) & ~X.index.isin(empty)].tolist()
        if unknown:
            problems.append(
                f"{col}: unknown category in row(s) {unknown}; "
                f"allowed values are {known}")

    if problems:
        raise ValueError("Invalid input - " + "; ".join(problems))
    return X, warnings


def predict_churn(df, bundle=None):
    """Return the input with churn_probability, predicted_churn, risk_label."""
    bundle = bundle or load_bundle()
    X, warnings = validate_input(df, bundle)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    result = df.copy()
    result["churn_probability"] = proba.round(4)
    result["predicted_churn"] = ["Yes" if p >= bundle["threshold"] else "No"
                                 for p in proba]
    result["risk_label"] = [risk_label(p) for p in proba]
    result.attrs["warnings"] = warnings
    return result


def format_predictions(result):
    """Human-readable text. The limitations message is ALWAYS included."""
    lines = []
    for idx, row in result.iterrows():
        who = row["customer_id"] if "customer_id" in result.columns else f"row {idx}"
        lines.append(f"{who}: {row['churn_probability']:.0%} chance of churn "
                     f"-> {row['risk_label']} risk "
                     f"(flagged as churner: {row['predicted_churn']})")
    for w in result.attrs.get("warnings", []):
        lines.append(f"Note: {w}")
    lines += ["", LIMITATIONS]
    return "\n".join(lines)


DEMO_CUSTOMERS = pd.DataFrame([
    dict(customer_id="NEW_MONTHLY", tenure_months=1, contract="Month-to-month",
         payment_method="Electronic check", internet_plan="Fiber",
         add_on_services=0, monthly_charges=85.0, monthly_usage_gb=210.0,
         logins_per_month=3, support_calls=5),
    dict(customer_id="LONG_TWO_YEAR", tenure_months=60, contract="Two-year",
         payment_method="Credit card", internet_plan="DSL",
         add_on_services=2, monthly_charges=55.0, monthly_usage_gb=95.0,
         logins_per_month=20, support_calls=0),
    dict(customer_id="MID_ONE_YEAR", tenure_months=24, contract="One-year",
         payment_method="Bank transfer", internet_plan="DSL",
         add_on_services=1, monthly_charges=52.0, monthly_usage_gb=100.0,
         logins_per_month=12, support_calls=1),
    dict(customer_id="MID_MONTHLY", tenure_months=10, contract="Month-to-month",
         payment_method="Mailed check", internet_plan="Fiber",
         add_on_services=1, monthly_charges=80.0, monthly_usage_gb=190.0,
         logins_per_month=9, support_calls=2),
])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, help="CSV with customer fields")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "predictions.csv")
    args = parser.parse_args()

    data = pd.read_csv(args.input) if args.input else DEMO_CUSTOMERS
    try:
        result = predict_churn(data)
    except ValueError as err:
        raise SystemExit(f"Error: {err}\nNo prediction was made.")
    print(format_predictions(result))
    if args.input:
        result.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
