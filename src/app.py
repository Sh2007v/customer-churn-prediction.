"""Simple Streamlit interface for the churn model (optional).

Run from the project root:
    pip install streamlit
    streamlit run src/app.py
"""
import pandas as pd
import streamlit as st

from predict import LIMITATIONS, load_bundle, predict_churn
from preprocess import ROOT

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.title("Customer Churn Prediction")
st.caption("Built on sample (synthetic) data - for learning, not for real decisions.")

try:
    bundle = load_bundle()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

cats = bundle["categories"]
rng = bundle["numeric_ranges"]

tab_one, tab_high = st.tabs(["Predict one customer", "High-risk customers"])

with tab_one:
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=3)
        contract = st.selectbox("Contract", cats["contract"],
                                index=cats["contract"].index("Month-to-month"))
        payment = st.selectbox("Payment method", cats["payment_method"])
        plan = st.selectbox("Internet plan", cats["internet_plan"])
        add_ons = st.number_input("Add-on services", min_value=0, max_value=10, value=0)
    with c2:
        charges = st.number_input("Monthly charges", min_value=0.0, value=85.0, step=1.0)
        usage = st.number_input("Monthly usage (GB)", min_value=0.0, value=150.0, step=10.0)
        logins = st.number_input("Logins per month", min_value=0, value=5)
        calls = st.number_input("Support calls", min_value=0, value=3)

    if st.button("Predict churn risk", type="primary"):
        customer = pd.DataFrame([{
            "tenure_months": tenure, "contract": contract, "payment_method": payment,
            "internet_plan": plan, "add_on_services": add_ons, "monthly_charges": charges,
            "monthly_usage_gb": usage, "logins_per_month": logins, "support_calls": calls}])
        try:
            result = predict_churn(customer, bundle)
            res = result.iloc[0]
            st.metric("Estimated churn probability", f"{res.churn_probability:.0%}")
            st.progress(float(res.churn_probability))
            st.write(f"**Risk label:** {res.risk_label}")
            for w in result.attrs["warnings"]:
                st.warning(w)
        except ValueError as err:
            st.error(f"{err}\n\nNo prediction was made.")
        # shown with every prediction attempt
        st.info(LIMITATIONS)

with tab_high:
    path = ROOT / "results" / "high_risk_customers.csv"
    if path.exists():
        st.write("Held-out test customers labelled **High** risk "
                 "(sorted by estimated probability).")
        st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
    else:
        st.write("Run `python src/train_model.py` to create this list.")
    st.info(LIMITATIONS)
