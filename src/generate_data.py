"""Generate a SAMPLE (synthetic) customer churn dataset.

No real customer data is used. Churn is made to depend on the features
(short tenure, month-to-month contracts, many support calls, ... churn more)
with random noise added, and the overall churn rate is calibrated to ~26%.

A few missing values and duplicate rows are injected on purpose so the
cleaning step has something realistic to do.

Usage (from the project root):
    python src/generate_data.py
    python src/generate_data.py --rows 2000 --seed 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "customer_data.csv"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n_rows=1500, seed=42, target_churn_rate=0.26):
    rng = np.random.default_rng(seed)

    # ---- Account information --------------------------------------------
    contract = rng.choice(["Month-to-month", "One-year", "Two-year"],
                          size=n_rows, p=[0.55, 0.25, 0.20])
    tenure = np.empty(n_rows)
    # (gamma shape, scale, shift): longer contracts -> longer tenure
    for name, shape, scale, shift in [("Month-to-month", 1.8, 11, 0),
                                      ("One-year", 3.0, 10, 6),
                                      ("Two-year", 4.0, 10, 12)]:
        m = contract == name
        tenure[m] = rng.gamma(shape, scale, m.sum()) + shift
    tenure = np.clip(np.round(tenure), 0, 72).astype(int)

    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_rows, p=[0.35, 0.25, 0.25, 0.15])

    # ---- Service information --------------------------------------------
    internet_plan = rng.choice(["DSL", "Fiber", "No internet"], size=n_rows,
                               p=[0.35, 0.45, 0.20])
    fiber = internet_plan == "Fiber"
    no_net = internet_plan == "No internet"
    add_on_services = np.where(no_net, 0, rng.binomial(4, 0.35, n_rows))
    base_price = pd.Series(internet_plan).map(
        {"DSL": 45.0, "Fiber": 75.0, "No internet": 20.0}).to_numpy()
    monthly_charges = np.clip(
        base_price + 5 * add_on_services + rng.normal(0, 6, n_rows), 15, 130
    ).round(2)

    # ---- Usage information ----------------------------------------------
    monthly_usage_gb = np.where(
        no_net, 0.0,
        rng.lognormal(mean=np.where(fiber, 5.3, 4.6), sigma=0.5)).round(1)
    logins_per_month = rng.poisson(12, n_rows)
    support_calls = rng.poisson(np.where(fiber, 1.6, 1.0))

    # ---- Churn depends on the features + random variation ---------------
    contract_effect = pd.Series(contract).map(
        {"Month-to-month": 0.9, "One-year": -0.3, "Two-year": -1.1}).to_numpy()
    z = (contract_effect
         - 0.035 * tenure
         + 0.018 * (monthly_charges - 60)
         + 0.30 * support_calls
         - 0.05 * (logins_per_month - 12)
         + 0.50 * fiber
         + 0.45 * (payment_method == "Electronic check")
         - 0.15 * add_on_services
         + rng.normal(0, 0.8, n_rows))          # noise: data is not "too perfect"

    # find the intercept that gives the wanted overall churn rate
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_churn_rate:
            hi = mid
        else:
            lo = mid
    p_churn = _sigmoid(z + (lo + hi) / 2)
    churn = rng.random(n_rows) < p_churn

    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_rows + 1)],
        "tenure_months": tenure,
        "contract": contract,
        "payment_method": payment_method,
        "internet_plan": internet_plan,
        "add_on_services": add_on_services,
        "monthly_charges": monthly_charges,
        "monthly_usage_gb": monthly_usage_gb,
        "logins_per_month": logins_per_month,
        "support_calls": support_calls,
        "churn": np.where(churn, "Yes", "No"),
    })

    # ---- Messiness on purpose: missing values + duplicate rows ----------
    idx = rng.choice(n_rows, size=int(0.015 * n_rows), replace=False)
    df.loc[idx, "monthly_usage_gb"] = np.nan
    idx = rng.choice(n_rows, size=int(0.01 * n_rows), replace=False)
    df.loc[idx, "payment_method"] = np.nan
    dupes = df.sample(8, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=1500,
                        help="number of customers (500-2000 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    rate = (df["churn"] == "Yes").mean()
    print(f"Saved {len(df)} rows to {args.out}")
    print(f"Churn rate: {rate:.1%} (synthetic data - not real customers)")


if __name__ == "__main__":
    main()
