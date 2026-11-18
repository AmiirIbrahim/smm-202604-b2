import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at module import time. Streamlit
# re-runs this whole script top-to-bottom on every widget interaction, so
# @st.cache_resource keeps joblib.load() from re-reading the files from
# disk on every single click -- without it, every tab switch and every
# form submission would reload all four artifacts again.
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # regression_pipeline = joblib.load("lasso_regressor_for_sme_revenue.joblib")
    # classification_pipeline = joblib.load("svc_classifier_for_sme_credit_risk.joblib")
    # clustering_artifact = joblib.load("kmeans_mall_customer_segmentation.joblib")
    # arm_artifact = joblib.load("apriori_recommendation_rules.joblib")
    MODEL_DIR = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
    )

    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib")
    )

    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
    )

    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_bank_wealth_segmentation.joblib")
    )

    arm_artifact = joblib.load(
        os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
    )
    return regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact

regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact = load_artifacts()
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
CLUSTERING_FEATURES = clustering_artifact["feature_order"]
CLUSTER_REGION_MEDIANS = clustering_artifact["region_median_digital_score"]
CLUSTER_NAMES = clustering_artifact["segment_names"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

st.set_page_config(page_title="BBT 4206 Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("BBT 4206 -- Model Validation Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "SME Revenue Regressor", "SME Credit Risk Classifier",
    "Bank Wealth Segmenter", "E-commerce Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression
# -----------------------------------------------------------------------
with tab_regression:
    with st.form("regression_form"):
        sector = st.text_input("Sector", "Technology")
        county = st.text_input("County", "Nairobi")
        quarter = st.text_input("Quarter", "Q3")
        business_age_years = st.number_input("Business Age (years)", value=4.5)
        num_employees = st.number_input("Number of Employees", value=12)
        # These four categories are the exact set the pipeline's
        # OrdinalEncoder was fitted on.
        owner_education_level = st.selectbox(
            "Owner Education Level",
            ["Primary", "Secondary", "Tertiary", "Postgraduate"], index=1)
        monthly_marketing_spend_kes = st.number_input("Monthly Marketing Spend (KES)", value=45000)
        avg_daily_foot_traffic = st.number_input(
            "Avg Daily Foot Traffic (leave 0 if not applicable)", value=0)
        online_presence_score = st.number_input("Online Presence Score", value=78.0)
        loan_amount_kes = st.number_input("Loan Amount (KES)", value=250000)
        interest_rate_pct = st.number_input("Interest Rate (%)", value=14.5)
        inflation_rate_pct = st.number_input("Inflation Rate (%)", value=6.5)
        competitor_density = st.number_input("Competitor Density", value=5)
        customer_satisfaction_score = st.number_input("Customer Satisfaction Score", value=4.2)
        submitted = st.form_submit_button("Predict Revenue")

    if submitted:
        payload = {
            "sector": sector, "county": county, "quarter": quarter,
            "business_age_years": business_age_years, "num_employees": num_employees,
            "owner_education_level": owner_education_level,
            "monthly_marketing_spend_kes": monthly_marketing_spend_kes,
            "avg_daily_foot_traffic": avg_daily_foot_traffic or None,
            "online_presence_score": online_presence_score,
            "loan_amount_kes": loan_amount_kes, "interest_rate_pct": interest_rate_pct,
            "inflation_rate_pct": inflation_rate_pct,
            "competitor_density": competitor_density,
            "customer_satisfaction_score": customer_satisfaction_score,
        }
        input_row = pd.DataFrame([payload])
        predicted_log_revenue = regression_pipeline.predict(input_row)[0]
        predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
        st.success(f"Predicted monthly revenue: KES {predicted_revenue_kes:,.2f}")

# -----------------------------------------------------------------------
# 2. Classification
# -----------------------------------------------------------------------
with tab_classification:
    with st.form("classification_form"):
        sector2 = st.text_input("Sector", "Retail")
        county2 = st.text_input("County", "Nairobi")
        quarter2 = st.text_input("Quarter", "Q2")
        loan_purpose = st.text_input("Loan Purpose", "Working Capital")
        owner_education_level2 = st.selectbox(
            "Owner Education Level",
            ["Primary", "Secondary", "Tertiary", "Postgraduate"], index=1, key="oel2")
        business_age_years2 = st.number_input("Business Age (years)", value=2.1, key="bay2")
        num_employees2 = st.number_input("Number of Employees", value=4, key="ne2")
        monthly_revenue_kes = st.number_input("Monthly Revenue (KES)", value=850000)
        loan_amount_kes2 = st.number_input("Loan Amount (KES)", value=150000, key="lak2")
        loan_term_months = st.number_input("Loan Term (months)", value=12)
        interest_rate_pct2 = st.number_input("Interest Rate (%)", value=16.5, key="irp2")
        collateral_value_kes = st.number_input("Collateral Value (leave 0 if unsecured)", value=0)
        debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", value=0.62)
        credit_score = st.number_input("Credit Score", value=590)
        num_previous_loans = st.number_input("Number of Previous Loans", value=1)
        missed_payments_count = st.number_input("Missed Payments Count", value=3)
        online_presence_score2 = st.number_input("Online Presence Score", value=40.0, key="ops2")
        customer_satisfaction_score2 = st.number_input(
            "Customer Satisfaction Score", value=3.4, key="css2")
        competitor_density2 = st.number_input("Competitor Density", value=6, key="cd2")
        submitted2 = st.form_submit_button("Predict Credit Risk")

    if submitted2:
        payload = {
            "sector": sector2, "county": county2, "quarter": quarter2,
            "loan_purpose": loan_purpose, "owner_education_level": owner_education_level2,
            "business_age_years": business_age_years2, "num_employees": num_employees2,
            "monthly_revenue_kes": monthly_revenue_kes, "loan_amount_kes": loan_amount_kes2,
            "loan_term_months": loan_term_months, "interest_rate_pct": interest_rate_pct2,
            "collateral_value_kes": collateral_value_kes or None,
            "debt_to_income_ratio": debt_to_income_ratio, "credit_score": credit_score,
            "num_previous_loans": num_previous_loans,
            "missed_payments_count": missed_payments_count,
            "online_presence_score": online_presence_score2,
            "customer_satisfaction_score": customer_satisfaction_score2,
            "competitor_density": competitor_density2,
        }
        input_row = pd.DataFrame([payload])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
        st.success(f"Predicted credit risk category: {predicted_class}")
        st.write({str(label): round(float(prob), 4)
                  for label, prob in zip(class_labels, predicted_probabilities)})

# -----------------------------------------------------------------------
# 3. Clustering
# -----------------------------------------------------------------------
def predict_segment(account_balance_kes, transactions_per_month, years_as_customer,
                    digital_banking_usage_score, branch_region):
    account_balance = float(account_balance_kes)
    transactions = float(transactions_per_month)
    tenure = float(years_as_customer)
    digital_score = (
        None if digital_banking_usage_score is None
        else float(digital_banking_usage_score)
    )
    values = [account_balance, transactions, tenure]
    if digital_score is not None:
        values.append(digital_score)
    if not np.isfinite(values).all():
        raise ValueError("Inputs must be finite numbers.")
    if account_balance <= 0:
        raise ValueError("Account balance must be greater than zero.")
    if transactions < 0 or tenure < 0:
        raise ValueError("Transactions and customer tenure cannot be negative.")
    if digital_score is not None and not 0 <= digital_score <= 100:
        raise ValueError("Digital usage score must be between 0 and 100.")
    if digital_score is None:
        if branch_region not in CLUSTER_REGION_MEDIANS:
            raise ValueError("Choose a known branch region when the digital score is missing.")
        digital_score = float(CLUSTER_REGION_MEDIANS[branch_region])

    transformed_values = {
        "log10_account_balance": float(np.log10(account_balance)),
        "transactions_per_month": transactions,
        "years_as_customer": tenure,
        "digital_banking_usage_score": digital_score,
    }
    input_row = pd.DataFrame([{
        feature: transformed_values[feature] for feature in CLUSTERING_FEATURES
    }])
    scaled_input = clustering_scaler.transform(input_row)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    return cluster, CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")


with tab_clustering:
    with st.form("clustering_form"):
        account_balance_kes = st.number_input(
            "Account Balance (KES)", min_value=1.0, value=2400000.0, step=1000.0)
        transactions_per_month = st.number_input(
            "Transactions per Month", min_value=0.0, value=9.0, step=1.0)
        years_as_customer = st.number_input(
            "Years as Customer", min_value=0.0, value=11.0, step=0.5)
        region_options = sorted(CLUSTER_REGION_MEDIANS)
        region_index = region_options.index("Nairobi") if "Nairobi" in region_options else 0
        branch_region = st.selectbox("Branch Region", region_options, index=region_index)
        digital_score_missing = st.checkbox("Digital usage score is missing")
        digital_banking_usage_score = None
        if not digital_score_missing:
            digital_banking_usage_score = st.number_input(
                "Digital Banking Usage Score", min_value=0.0, max_value=100.0, value=55.0)
        submitted3 = st.form_submit_button("Predict Segment")

    if submitted3:
        try:
            cluster, cluster_name = predict_segment(
                account_balance_kes,
                transactions_per_month,
                years_as_customer,
                digital_banking_usage_score,
                branch_region,
            )
            st.success(f"{cluster_name} (cluster {cluster})")
        except (TypeError, ValueError, OverflowError) as error:
            st.error(str(error))

# -----------------------------------------------------------------------
# 4. Association Rule Mining
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        cart_text = st.text_input("Cart (comma-separated SKUs)", "H06, H07")
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4:
        cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
        candidates = {}
        for antecedent, consequent_list in recommendation_lookup.items():
            if antecedent.issubset(cart):
                for sku, confidence, lift in consequent_list:
                    if sku not in cart:
                        if sku not in candidates or confidence > candidates[sku][0]:
                            candidates[sku] = (confidence, lift)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
        if not ranked:
            st.warning("No recommendations for this cart -- try a SKU combination seen in the training orders.")
        else:
            for sku, (confidence, lift) in ranked:
                st.write(f"**{sku_lookup.loc[sku, 'product_name']}** ({sku}) "
                         f"-- confidence {confidence:.1%}, lift {lift:.2f}")