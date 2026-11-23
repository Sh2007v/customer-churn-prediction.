"""Run the model/code test cases and record them in results/evaluation.md.

Usage (after training):  python src/run_tests.py
Exit code is 1 if any test fails.
"""
import sys

import pandas as pd

from predict import (DEMO_CUSTOMERS, LIMITATIONS, format_predictions,
                     load_bundle, predict_churn)
from preprocess import ROOT

RESULTS = ROOT / "results"
bundle = load_bundle()

NEW_MONTHLY = DEMO_CUSTOMERS.iloc[[0]].reset_index(drop=True)
LONG_TWO_YEAR = DEMO_CUSTOMERS.iloc[[1]].reset_index(drop=True)

rows = []


def record(tc_id, scenario, expected, check):
    """check() returns (passed: bool, actual: str)."""
    try:
        ok, actual = check()
    except Exception as err:  # an unexpected crash is a failure
        ok, actual = False, f"Unexpected error: {err}"
    rows.append((tc_id, scenario, expected, actual, "Pass" if ok else "Fail"))


def expect_error(df, must_contain):
    try:
        predict_churn(df, bundle)
    except ValueError as err:
        return must_contain.lower() in str(err).lower(), f"ValueError: {err}"
    return False, "A prediction was returned for invalid input"


def tc01():
    r = predict_churn(NEW_MONTHLY, bundle).iloc[0]
    return (r.risk_label == "High", f"{r.churn_probability:.0%} -> {r.risk_label} risk")


def tc02():
    r = predict_churn(LONG_TWO_YEAR, bundle).iloc[0]
    return (r.risk_label == "Low", f"{r.churn_probability:.0%} -> {r.risk_label} risk")


def tc03():
    bad = NEW_MONTHLY.copy()
    bad["monthly_charges"] = -50.0
    return expect_error(bad, "negative")


def tc04():
    text = format_predictions(predict_churn(DEMO_CUSTOMERS, bundle))
    app_src = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    ok = LIMITATIONS in text and "LIMITATIONS" in app_src
    return ok, "Message present in text/CLI output and used by the app"


def tc05():
    ext = NEW_MONTHLY.copy()
    ext["tenure_months"], ext["monthly_charges"] = 0, 5000.0
    r = predict_churn(ext, bundle)
    p = r.churn_probability.iloc[0]
    warned = len(r.attrs["warnings"]) > 0
    return (0 <= p <= 1 and warned,
            f"Probability {p:.0%}; out-of-range warning shown: {warned}")


def tc06():
    return expect_error(NEW_MONTHLY.drop(columns=["support_calls"]), "support_calls")


def tc07():
    bad = NEW_MONTHLY.copy()
    bad["support_calls"] = None
    return expect_error(bad, "support_calls")


def tc08():
    bad = NEW_MONTHLY.copy()
    bad["contract"] = "Lifetime"
    return expect_error(bad, "unknown category")


def tc09():
    p = pd.read_csv(RESULTS / "test_predictions.csv")["churn_probability"]
    return (p.between(0, 1).all(), f"{len(p)} held-out predictions, "
            f"min {p.min():.3f}, max {p.max():.3f}")


def tc10():
    base = NEW_MONTHLY.copy()
    two = base.copy()
    two["contract"] = "Two-year"
    p_m = predict_churn(base, bundle).churn_probability.iloc[0]
    p_t = predict_churn(two, bundle).churn_probability.iloc[0]
    return p_m > p_t, f"Monthly {p_m:.0%} vs two-year {p_t:.0%} (all else equal)"


record("TC_CHURN_01", "New customer, monthly contract, many support calls",
       "Higher churn probability and High risk label", tc01)
record("TC_CHURN_02", "Long-term customer on a two-year contract",
       "Lower churn probability and Low risk label", tc02)
record("TC_CHURN_03", "Negative value for monthly charges",
       "Clear error message, no prediction", tc03)
record("TC_CHURN_04", "Limitations message appears with every prediction",
       "Message is visible", tc04)
record("TC_CHURN_05", "Extreme values: zero tenure, monthly charges of 5000",
       "Probability between 0 and 1, plus a reliability warning", tc05)
record("TC_CHURN_06", "Missing field (column removed)",
       "Clear error naming the field, no prediction", tc06)
record("TC_CHURN_07", "Missing value in a required field",
       "Clear error naming the field, no prediction", tc07)
record("TC_CHURN_08", "Unknown category (contract = 'Lifetime')",
       "Clear error listing allowed values, no prediction", tc08)
record("TC_CHURN_09", "All held-out test predictions",
       "Every probability is between 0 and 1", tc09)
record("TC_CHURN_10", "Same customer on monthly vs two-year contract",
       "Monthly contract has the higher churn probability", tc10)

# ---- write results --------------------------------------------------------
md = ["| Test case | Scenario | Expected | Actual | Result |", "|---|---|---|---|---|"]
for tc_id, scenario, expected, actual, result in rows:
    actual = actual.replace("|", "/").replace("\n", " ")
    md.append(f"| {tc_id} | {scenario} | {expected} | {actual} | **{result}** |")
table = "\n".join(md)

(RESULTS / "test_results.md").write_text("# Test results\n\n" + table + "\n", encoding="utf-8")
ev = RESULTS / "evaluation.md"
text = ev.read_text(encoding="utf-8")
start, end = "<!-- TESTS START -->", "<!-- TESTS END -->"
if start in text and end in text:
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    text = head + start + "\n" + table + "\n" + end + tail
else:
    text += f"\n## Test cases\n\n{start}\n{table}\n{end}\n"
ev.write_text(text, encoding="utf-8")

failed = [r for r in rows if r[4] == "Fail"]
for r in rows:
    print(f"{r[0]}: {r[4]} - {r[3]}")
print(f"\n{len(rows) - len(failed)}/{len(rows)} tests passed. Recorded in results/evaluation.md")
sys.exit(1 if failed else 0)
