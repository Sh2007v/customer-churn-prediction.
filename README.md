# Customer Churn Prediction

A beginner-to-intermediate machine learning project that estimates **which customers may stop using a service (churn)** so a business can act early. It covers data preparation, classification with scikit-learn, evaluation beyond simple accuracy (imbalanced data), and responsible reporting of predictions.

> **Important - sample data only.** `data/customer_data.csv` is **synthetic (sample) data** created by `src/generate_data.py`. It contains **no real customer personal data**. Churn was generated from simple rules (short tenure, monthly contracts, many support calls, etc. churn more) **plus random noise** so the data is not "too perfect". Results will not match real-world performance.

## Project structure

```
customer-churn-prediction/
├── data/
│   └── customer_data.csv        # synthetic sample dataset (1,508 raw rows)
├── notebooks/
│   └── churn_prediction.ipynb   # exploration, training and evaluation
├── src/
│   ├── generate_data.py         # creates the sample dataset
│   ├── preprocess.py            # cleaning, encoding, scaling, train/test split
│   ├── train_model.py           # trains, compares and evaluates models; saves charts + results
│   ├── predict.py               # predicts churn for new customers (with input validation)
│   ├── run_tests.py             # runs the test cases and records them in evaluation.md
│   ├── plots.py                 # chart helpers
│   └── app.py                   # simple Streamlit interface (optional)
├── models/
│   └── churn_model.pkl          # saved trained model (pipeline + threshold)
├── results/
│   ├── evaluation.md            # metrics, observations, test results, limitations
│   ├── test_predictions.csv     # churn probability for every held-out test customer
│   ├── high_risk_customers.csv  # held-out customers labelled "High" risk
│   ├── metrics.json
│   └── charts/                  # confusion matrix, ROC, feature importance, EDA charts
├── requirements.txt
└── README.md
```

## Dataset (synthetic)

| Column | Meaning |
|---|---|
| `customer_id` | Identifier (ignored by the model) |
| `tenure_months` | How long the customer has been with the service |
| `contract` | Month-to-month, One-year, Two-year |
| `payment_method` | Electronic check, Credit card, Bank transfer, Mailed check |
| `internet_plan` | DSL, Fiber, No internet |
| `add_on_services` | Number of add-ons (0-4) |
| `monthly_charges` | Monthly bill |
| `monthly_usage_gb` | Monthly data usage |
| `logins_per_month` | Number of logins |
| `support_calls` | Number of support calls |
| `churn` | **Yes** (customer left) / **No** (customer stayed) - the target |

About 25% of customers churn. The file deliberately contains a few missing values and 8 duplicate rows so the cleaning step has real work to do.

## How to run

```bash
pip install -r requirements.txt

python src/generate_data.py      # 1. create the sample dataset (already included)
python src/train_model.py        # 2. clean, split, train, evaluate, save model + charts
python src/run_tests.py          # 3. run test cases (results go into results/evaluation.md)
python src/predict.py            # 4. predict 4 demo customers
python src/predict.py --input my_customers.csv    # or your own CSV

streamlit run src/app.py         # optional web interface
jupyter notebook notebooks/churn_prediction.ipynb # optional walkthrough
```

Run all commands from the project root. Useful options: `python src/train_model.py --final "Random Forest"` or `--threshold 0.4`.

## Method

1. **Clean** - remove duplicates and impossible values (e.g., negative charges); ignore the ID column.
2. **Preprocess** - median-impute and scale numbers (`StandardScaler`); impute and one-hot encode categories. All of this is inside a scikit-learn `Pipeline`, so it is **fitted on training data only** (no leakage).
3. **Split** - 80% train / 20% test, `stratify=y`, `random_state=42`.
4. **Train** - Logistic Regression (final model, simple and explainable), plus Decision Tree and Random Forest for comparison. `class_weight="balanced"` handles the imbalance. (SMOTE was not used; see *Possible next steps*.)
5. **Choose the decision threshold** using cross-validated predictions on the *training* data (best F1), so the test set stays untouched. Chosen: **0.55**.
6. **Evaluate** on the test set with accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.

## Results (test set, 300 customers)

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (final, threshold 0.50) | 74.3% | 0.49 | 0.73 | 0.59 | 0.79 |
| Logistic Regression (final, threshold 0.55) | 76.0% | 0.51 | 0.71 | 0.60 | 0.79 |
| Decision Tree | 66.0% | 0.40 | 0.71 | 0.51 | 0.74 |
| Random Forest | 73.0% | 0.47 | 0.71 | 0.57 | 0.78 |

At threshold 0.55 the final model finds about 71% of churners and about half of the customers it flags really churn (53 found, 22 missed, 50 false alarms). Always predicting "No churn" would already be 75% accurate while finding **no** churners, which is why accuracy alone is not enough here. Full details, the confusion matrix and charts: `results/evaluation.md` and `results/charts/`.

**What drives churn in this data:** monthly contracts, higher monthly charges and more support calls push risk up; two-year contracts and longer tenure push it down. This is a pattern in the (synthetic) data, **not proof of cause**.

## Sample prediction

```
NEW_MONTHLY:   98% chance of churn -> High risk
LONG_TWO_YEAR:  1% chance of churn -> Low risk

These are estimates based on sample data and past patterns. They are not certain, and some predictions will be wrong.
```

Risk labels: **Low** < 30%, **Medium** 30-60%, **High** >= 60%. Because of the balanced class weights, probabilities are *inflated* compared with the true churn rate - use them to **rank** customers by risk rather than as exact chances.

## Testing

`python src/run_tests.py` runs 10 test cases: typical customers (new monthly vs long-term two-year), extreme values (zero tenure, very high charges), invalid input (negative values, missing field, missing value, unknown category), probabilities staying within 0-1, the limitations message being shown, and a sanity check that a monthly contract scores riskier than a two-year one. All 10 pass; results are recorded in `results/evaluation.md`.

## Limitations

- Predictions are estimates based on patterns in sample data, not facts about any real person.
- Real customers leave for many reasons the data may not show (price changes, competitors, personal reasons).
- The sample data is simpler than real business data, so results will not match real-world performance.
- Some customers will be wrongly flagged (false positives) and some churners will be missed (false negatives).
- Predictions should support human decisions, never replace them.
- Use customer data responsibly and avoid unfair treatment based on sensitive personal details. Never use real personal customer data without permission.

## Possible next steps

- Try SMOTE (`imbalanced-learn`, on training data only) and compare with class weights.
- Calibrate probabilities (e.g., `CalibratedClassifierCV`) so they match real churn rates.
- Try the public Telco Customer Churn dataset (Kaggle) for practice with real-world-style data.
- Choose the threshold from real costs: missing a churner vs. a wasted retention offer.
