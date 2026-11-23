# Evaluation - Customer Churn Prediction

> All results come from **sample (synthetic) data**. Real-world performance will differ.

## 1. Dataset

- Rows after cleaning: 1500 (from 1508 raw rows)
- Duplicates removed: 8; impossible rows removed: 0; missing values found (imputed later): {'monthly_usage_gb': 22, 'payment_method': 15}
- Churn rate: 24.9% (imbalanced: most customers stay)
- Train/test split: 1200 / 300 rows (80/20, stratified, random_state=42)
- Baseline: always predicting "No churn" would be 75.0% accurate on the test set but would find **0%** of churners, so accuracy alone is misleading.

## 2. Model comparison (test set, threshold 0.50)

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC | CV ROC-AUC (train) |
|---|---|---|---|---|---|---|
| Logistic Regression (final) | 74.3% | 0.49 | 0.73 | 0.59 | 0.79 | 0.79 ± 0.04 |
| Decision Tree | 66.0% | 0.40 | 0.71 | 0.51 | 0.74 | 0.73 ± 0.04 |
| Random Forest | 73.0% | 0.47 | 0.71 | 0.57 | 0.78 | 0.77 ± 0.05 |

## 3. Final model: Logistic Regression

Why: it is simple and easy to explain. Decision threshold chosen on cross-validated training predictions (best F1): **0.55**.

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.50 | 74.3% | 0.49 | 0.73 | 0.59 | 0.79 |
| 0.55 | 76.0% | 0.51 | 0.71 | 0.60 | 0.79 |

Confusion matrix at threshold 0.55 (test set):

| | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | 175 (true negatives) | 50 (false alarms) |
| **Actually churned** | 22 (missed churners) | 53 (found churners) |

Charts are in `results/charts/`: `confusion_matrix.png`, `roc_curves.png`, `feature_importance.png`, `threshold_tradeoff.png`, and the data exploration charts.

## 4. Most influential features

Logistic Regression coefficients (numeric features are scaled, so sizes are comparable).

- Push churn risk **up**: contract_Month-to-month, monthly_charges, support_calls
- Push churn risk **down**: contract_Two-year, tenure_months, payment_method_Credit card

In plain words: in this sample data, customers on monthly contracts with short tenure and more support calls are more likely to churn. This is a pattern in the data, not proof of cause.

## 5. Observations

- At threshold 0.55 the model finds 70.7% of churners (recall) and 51.5% of the customers it flags really churn (precision). 22 churners were missed and 50 customers were flagged wrongly.
- Highest test ROC-AUC: Logistic Regression (0.79). Differences between models are small on a dataset this size, so treat the ranking with care.
- Lowering the threshold catches more churners (higher recall) but flags more customers who would have stayed (lower precision). The right threshold depends on whether a missed churner or a false alarm costs more.
- Because class weights are balanced, the probabilities are **inflated** compared with the true churn rate. They are best used to **rank** customers by risk, not as exact chances.
- A pattern in the data does not prove the cause of churn.

## 6. Test cases

<!-- TESTS START -->
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
<!-- TESTS END -->

## 7. Limitations

> These are estimates based on sample data and past patterns. They are not certain, and some predictions will be wrong.

- Predictions are estimates based on patterns in sample data, not facts about any real person.
- Real customers leave for reasons the data may not show (price changes, competitors, personal reasons).
- The sample data is much simpler than real business data (it was generated from a few rules plus noise), so results will not match real-world performance.
- Some customers will be wrongly flagged (false positives) and some churners will be missed (false negatives).
- Predictions should support human decisions, never replace them.
- Use customer data responsibly and avoid unfair treatment based on sensitive personal details.
