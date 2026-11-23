"""Train, compare and evaluate churn models; save the final model, charts and
results/evaluation.md.

Usage (from the project root):
    python src/train_model.py                    # final model: Logistic Regression
    python src/train_model.py --final "Random Forest"
    python src/train_model.py --threshold 0.4    # override the decision threshold
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import plots
import preprocess as pp
from predict import LIMITATIONS, MODEL_PATH, risk_label

RESULTS = pp.ROOT / "results"
CHARTS = RESULTS / "charts"
RS = pp.RANDOM_STATE


def make_models():
    # class_weight="balanced": churners are the minority, so they get more weight
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RS),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced",
            random_state=RS),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RS, n_jobs=-1),
    }


def make_pipeline(model):
    # preprocessing is INSIDE the pipeline -> fitted on training data only
    return Pipeline([("prep", pp.build_preprocessor()), ("model", model)])


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def choose_threshold(pipeline, X_train, y_train, cv):
    """Pick the threshold with the best F1 using cross-validated predictions
    on TRAINING data only (the test set stays untouched)."""
    proba = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                              method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.20, 0.801, 0.05), 2)
    prec, rec, f1 = [], [], []
    for t in grid:
        pred = (proba >= t).astype(int)
        prec.append(precision_score(y_train, pred, zero_division=0))
        rec.append(recall_score(y_train, pred, zero_division=0))
        f1.append(f1_score(y_train, pred, zero_division=0))
    best = float(grid[int(np.argmax(f1))])
    return best, grid, prec, rec, f1


def feature_effects(fitted_pipeline):
    names = [n.split("__", 1)[1] for n in
             fitted_pipeline.named_steps["prep"].get_feature_names_out()]
    model = fitted_pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        return names, model.coef_[0], True
    return names, model.feature_importances_, False


def pct(x):
    return f"{x:.1%}"


def write_evaluation(path, ctx):
    rep, m = ctx["report"], ctx["models"]
    final, fm_default, fm_chosen = ctx["final"], ctx["final_default"], ctx["final_chosen"]
    cm = fm_chosen["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    L = []
    L.append("# Evaluation - Customer Churn Prediction\n")
    L.append("> All results come from **sample (synthetic) data**. "
             "Real-world performance will differ.\n")
    L.append("## 1. Dataset\n")
    L.append(f"- Rows after cleaning: {rep['rows_out']} (from {rep['rows_in']} raw rows)")
    L.append(f"- Duplicates removed: {rep['duplicates_removed']}; "
             f"impossible rows removed: {rep['impossible_rows_removed']}; "
             f"missing values found (imputed later): {rep['missing_values_found']}")
    L.append(f"- Churn rate: {pct(ctx['churn_rate'])} "
             "(imbalanced: most customers stay)")
    L.append(f"- Train/test split: {ctx['n_train']} / {ctx['n_test']} rows "
             "(80/20, stratified, random_state=42)")
    L.append(f"- Baseline: always predicting \"No churn\" would be "
             f"{pct(1 - ctx['test_churn_rate'])} accurate on the test set "
             "but would find **0%** of churners, so accuracy alone is misleading.\n")

    L.append("## 2. Model comparison (test set, threshold 0.50)\n")
    L.append("| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC | CV ROC-AUC (train) |")
    L.append("|---|---|---|---|---|---|---|")
    for name, r in m.items():
        e = r["test_default"]
        L.append(f"| {name}{' (final)' if name == final else ''} | {pct(e['accuracy'])} | "
                 f"{e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} | "
                 f"{e['roc_auc']:.2f} | {r['cv_auc_mean']:.2f} ± {r['cv_auc_std']:.2f} |")
    L.append("")

    L.append(f"## 3. Final model: {final}\n")
    L.append(f"Why: it is simple and easy to explain. Decision threshold chosen on "
             f"cross-validated training predictions (best F1): **{fm_chosen['threshold']:.2f}**"
             f"{' (manual override)' if ctx['override'] else ''}.\n")
    L.append("| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    L.append("|---|---|---|---|---|---|")
    for e in (fm_default, fm_chosen):
        L.append(f"| {e['threshold']:.2f} | {pct(e['accuracy'])} | {e['precision']:.2f} | "
                 f"{e['recall']:.2f} | {e['f1']:.2f} | {e['roc_auc']:.2f} |")
    L.append("")
    L.append(f"Confusion matrix at threshold {fm_chosen['threshold']:.2f} (test set):\n")
    L.append("| | Predicted stay | Predicted churn |")
    L.append("|---|---|---|")
    L.append(f"| **Actually stayed** | {tn} (true negatives) | {fp} (false alarms) |")
    L.append(f"| **Actually churned** | {fn} (missed churners) | {tp} (found churners) |")
    L.append("")
    L.append("Charts are in `results/charts/`: `confusion_matrix.png`, `roc_curves.png`, "
             "`feature_importance.png`, `threshold_tradeoff.png`, and the data "
             "exploration charts.\n")

    L.append("## 4. Most influential features\n")
    L.append(ctx["feature_text"] + "\n")

    L.append("## 5. Observations\n")
    L.append(f"- At threshold {fm_chosen['threshold']:.2f} the model finds "
             f"{pct(fm_chosen['recall'])} of churners (recall) and "
             f"{pct(fm_chosen['precision'])} of the customers it flags really churn "
             f"(precision). {fn} churners were missed and {fp} customers were flagged "
             "wrongly.")
    best_auc = max(m, key=lambda k: m[k]["test_default"]["roc_auc"])
    L.append(f"- Highest test ROC-AUC: {best_auc} "
             f"({m[best_auc]['test_default']['roc_auc']:.2f}). Differences between "
             "models are small on a dataset this size, so treat the ranking with care.")
    L.append("- Lowering the threshold catches more churners (higher recall) but flags "
             "more customers who would have stayed (lower precision). The right "
             "threshold depends on whether a missed churner or a false alarm costs more.")
    L.append("- Because class weights are balanced, the probabilities are **inflated** "
             "compared with the true churn rate. They are best used to **rank** "
             "customers by risk, not as exact chances.")
    L.append("- A pattern in the data does not prove the cause of churn.\n")

    L.append("## 6. Test cases\n")
    L.append("<!-- TESTS START -->")
    L.append("_Run `python src/run_tests.py` to fill this section._")
    L.append("<!-- TESTS END -->\n")

    L.append("## 7. Limitations\n")
    L.append(f"> {LIMITATIONS}\n")
    for t in [
        "Predictions are estimates based on patterns in sample data, not facts about any real person.",
        "Real customers leave for reasons the data may not show (price changes, competitors, personal reasons).",
        "The sample data is much simpler than real business data (it was generated from a few rules plus noise), so results will not match real-world performance.",
        "Some customers will be wrongly flagged (false positives) and some churners will be missed (false negatives).",
        "Predictions should support human decisions, never replace them.",
        "Use customer data responsibly and avoid unfair treatment based on sensitive personal details.",
    ]:
        L.append(f"- {t}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--final", default="Logistic Regression",
                        choices=list(make_models()))
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the automatically chosen threshold")
    args = parser.parse_args()
    CHARTS.mkdir(parents=True, exist_ok=True)

    # 1. load + clean
    raw = pp.load_data()
    df, report = pp.clean_data(raw)
    print("Cleaning report:", report)

    # 2. exploration charts
    plots.plot_class_distribution(df["churn"], CHARTS / "class_distribution.png")
    plots.plot_churn_rate_by(df, "contract", CHARTS / "churn_by_contract.png",
                             title="Churn rate by contract type")
    plots.plot_churn_rate_by(df, "tenure_months", CHARTS / "churn_by_tenure.png",
                             bins=[0, 6, 12, 24, 48, 72],
                             title="Churn rate by tenure (months)")
    plots.plot_churn_rate_by(df, "monthly_charges", CHARTS / "churn_by_monthly_charges.png",
                             bins=[0, 40, 60, 80, 100, 130],
                             title="Churn rate by monthly charges")

    # 3. split
    X_train, X_test, y_train, y_test = pp.split_data(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    # 4. train + compare
    results, fitted, curves = {}, {}, {}
    for name, model in make_models().items():
        pipe = make_pipeline(model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        results[name] = {"test_default": evaluate(y_test, proba, 0.5),
                         "cv_auc_mean": float(cv_auc.mean()),
                         "cv_auc_std": float(cv_auc.std())}
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr, results[name]["test_default"]["roc_auc"])
        e = results[name]["test_default"]
        print(f"{name:20s} acc={e['accuracy']:.3f} prec={e['precision']:.2f} "
              f"rec={e['recall']:.2f} f1={e['f1']:.2f} auc={e['roc_auc']:.3f}")

    # 5. final model + threshold (chosen WITHOUT looking at the test set)
    final_name = args.final
    final_pipe = fitted[final_name]
    best_t, grid, prec, rec, f1 = choose_threshold(
        make_pipeline(make_models()[final_name]), X_train, y_train, cv)
    threshold = args.threshold if args.threshold is not None else best_t
    proba_test = final_pipe.predict_proba(X_test)[:, 1]
    final_default = evaluate(y_test, proba_test, 0.5)
    final_chosen = evaluate(y_test, proba_test, threshold)
    print(f"Final model: {final_name}, threshold={threshold:.2f} -> "
          f"recall={final_chosen['recall']:.2f}, precision={final_chosen['precision']:.2f}")

    # 6. charts for the final model
    plots.plot_confusion_matrix(final_chosen["confusion_matrix"], CHARTS / "confusion_matrix.png",
                                f"{final_name}\nthreshold {threshold:.2f} (test set)")
    plots.plot_roc_curves(curves, CHARTS / "roc_curves.png")
    plots.plot_threshold_tradeoff(grid, prec, rec, f1, threshold, CHARTS / "threshold_tradeoff.png")
    names, values, signed = feature_effects(final_pipe)
    plots.plot_feature_importance(
        names, values, CHARTS / "feature_importance.png",
        f"{final_name}: " + ("coefficients (scaled features)" if signed else "feature importance"),
        signed=signed)

    # plain-language feature summary
    order = np.argsort(values)
    if signed:
        up = [names[i] for i in order[::-1][:3]]
        down = [names[i] for i in order[:3]]
        feature_text = ("Logistic Regression coefficients (numeric features are scaled, "
                        "so sizes are comparable).\n\n"
                        f"- Push churn risk **up**: {', '.join(up)}\n"
                        f"- Push churn risk **down**: {', '.join(down)}\n\n"
                        "In plain words: in this sample data, customers on monthly contracts "
                        "with short tenure and more support calls are more likely to churn. "
                        "This is a pattern in the data, not proof of cause.")
    else:
        top = [names[i] for i in np.argsort(values)[::-1][:5]]
        feature_text = f"Top features by importance: {', '.join(top)}."

    # 7. predictions for held-out test customers (never seen in training)
    test_pred = pd.DataFrame({
        "customer_id": df.loc[X_test.index, pp.ID_COLUMN].values,
        "actual_churn": np.where(y_test.values == 1, "Yes", "No"),
        "churn_probability": proba_test.round(4),
        "predicted_churn": np.where(proba_test >= threshold, "Yes", "No"),
        "risk_label": [risk_label(p) for p in proba_test],
    }).sort_values("churn_probability", ascending=False)
    test_pred.to_csv(RESULTS / "test_predictions.csv", index=False)
    high = test_pred[test_pred["risk_label"] == "High"]
    high.to_csv(RESULTS / "high_risk_customers.csv", index=False)
    print(f"Saved test predictions ({len(test_pred)}) and {len(high)} high-risk customers.")

    # 8. save model bundle
    X_all = df[pp.FEATURES]
    bundle = {
        "pipeline": final_pipe,
        "model_name": final_name,
        "threshold": float(threshold),
        "categories": {c: sorted(X_all[c].dropna().unique().tolist()) for c in pp.CATEGORICAL},
        "numeric_ranges": {c: (float(X_all[c].min()), float(X_all[c].max())) for c in pp.NUMERIC},
        "metrics": final_chosen,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    ctx = {"report": report, "models": results, "final": final_name,
           "final_default": final_default, "final_chosen": final_chosen,
           "override": args.threshold is not None,
           "churn_rate": float(df["churn"].mean()),
           "test_churn_rate": float(y_test.mean()),
           "n_train": len(X_train), "n_test": len(X_test),
           "feature_text": feature_text}
    write_evaluation(RESULTS / "evaluation.md", ctx)
    (RESULTS / "metrics.json").write_text(json.dumps(
        {"models": results, "final_model": final_name, "final_at_chosen_threshold": final_chosen,
         "final_at_0.5": final_default}, indent=2))
    print("Saved model, charts, results/evaluation.md")


if __name__ == "__main__":
    main()
