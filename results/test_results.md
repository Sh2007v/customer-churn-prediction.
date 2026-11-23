# Test results

| Test case | Scenario | Expected | Actual | Result |
|---|---|---|---|---|
| TC_CHURN_01 | New customer, monthly contract, many support calls | Higher churn probability and High risk label | 98% -> High risk | **Pass** |
| TC_CHURN_02 | Long-term customer on a two-year contract | Lower churn probability and Low risk label | 1% -> Low risk | **Pass** |
| TC_CHURN_03 | Negative value for monthly charges | Clear error message, no prediction | ValueError: Invalid input - monthly_charges: negative value not allowed in row(s) [0] | **Pass** |
| TC_CHURN_04 | Limitations message appears with every prediction | Message is visible | Message present in text/CLI output and used by the app | **Pass** |
| TC_CHURN_05 | Extreme values: zero tenure, monthly charges of 5000 | Probability between 0 and 1, plus a reliability warning | Probability 100%; out-of-range warning shown: True | **Pass** |
| TC_CHURN_06 | Missing field (column removed) | Clear error naming the field, no prediction | ValueError: Missing required field(s): support_calls | **Pass** |
| TC_CHURN_07 | Missing value in a required field | Clear error naming the field, no prediction | ValueError: Invalid input - support_calls: missing or non-numeric value in row(s) [0] | **Pass** |
| TC_CHURN_08 | Unknown category (contract = 'Lifetime') | Clear error listing allowed values, no prediction | ValueError: Invalid input - contract: unknown category in row(s) [0]; allowed values are ['Month-to-month', 'One-year', 'Two-year'] | **Pass** |
| TC_CHURN_09 | All held-out test predictions | Every probability is between 0 and 1 | 300 held-out predictions, min 0.009, max 0.927 | **Pass** |
| TC_CHURN_10 | Same customer on monthly vs two-year contract | Monthly contract has the higher churn probability | Monthly 98% vs two-year 86% (all else equal) | **Pass** |
