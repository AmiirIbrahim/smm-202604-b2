import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="BBT 4206 Model Demos", page_icon="\U0001F4CA", layout="wide")

# -----------------------------------------------------------------------
# Load the same merged artifacts as api.py. Streamlit reruns this script
# after interactions, so cache the fitted objects once per process.
# -----------------------------------------------------------------------
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
)


@st.cache_resource
def load_artifacts():
    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "student_exam_score_regressor.joblib")
    )
    fraud_artifact = joblib.load(os.path.join(MODEL_DIR, "fraud_model_bundle.joblib"))
    arm_artifact = joblib.load(os.path.join(MODEL_DIR, "arm_pharmacy_basket_rules.joblib"))
    return regression_pipeline, fraud_artifact, arm_artifact


regression_pipeline, fraud_artifact, arm_artifact = load_artifacts()
classification_pipeline = fraud_artifact["model"]
classification_threshold = float(fraud_artifact["threshold"])
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

REGRESSION_DEFAULTS = {
    "major_subject": "Engineering", "parental_education_level": "Bachelor's",
    "internet_access_quality": "Good", "part_time_job": "No",
    "study_group_participation": "Yes", "study_hours_per_week": 9.5,
    "attendance_rate_pct": 88.0, "previous_gpa": 3.4,
    "sleep_hours_per_night": 7.0, "extracurricular_activities_count": 1,
    "stress_level_score": 5.0, "tutoring_hours_per_week": 2.0,
}
REGRESSION_FIELDS = list(regression_pipeline.feature_names_in_)
CLASSIFICATION_DEFAULTS = {
    "policy_type": "Comprehensive", "claim_type": "Collision",
    "incident_location_type": "Highway", "witnesses_present": "No",
    "agent_flagged_history": "Yes", "police_report_filed": "No",
    "claimant_age": 34, "policy_tenure_years": 1.2,
    "prior_claims_count": 3, "days_to_report_claim": 18,
    "claim_amount_kes": 480000.0,
    "claim_amount_to_policy_limit_ratio": 0.92, "customer_credit_score": 540.0,
}
CLASSIFICATION_FIELDS = list(fraud_artifact["features"])

st.title("BBT 4206 Model Demos")
st.caption("Live model validation using the team's saved artifacts")

tab_regression, tab_classification, tab_clustering, tab_recommender = st.tabs([
    "Exam score", "Fraud classification", "Clustering · pending", "Pharmacy recommender",
])


def form_inputs(fields, defaults, prefix):
    values = {}
    columns = st.columns(2)
    for index, field in enumerate(fields):
        default = defaults[field]
        with columns[index % 2]:
            if isinstance(default, str):
                values[field] = st.text_input(field.replace("_", " ").title(), default, key=f"{prefix}_{field}")
            else:
                step = 1 if isinstance(default, int) else 0.1
                values[field] = st.number_input(
                    field.replace("_", " ").title(), value=default, step=step,
                    key=f"{prefix}_{field}",
                )
    return values

# -----------------------------------------------------------------------
# 1. Regression
# -----------------------------------------------------------------------
with tab_regression:
    st.subheader("Student exam score")
    with st.form("exam_score_form"):
        regression_values = form_inputs(REGRESSION_FIELDS, REGRESSION_DEFAULTS, "exam")
        submitted_regression = st.form_submit_button("Predict exam score")
    if submitted_regression:
        try:
            input_row = pd.DataFrame([regression_values])[REGRESSION_FIELDS]
            prediction = float(regression_pipeline.predict(input_row)[0])
            st.success(f"Predicted final exam score: {prediction:.2f}")
        except (TypeError, ValueError, KeyError) as error:
            st.error(str(error))

with tab_classification:
    st.subheader("Insurance fraud classification")
    with st.form("fraud_form"):
        classification_values = form_inputs(
            CLASSIFICATION_FIELDS, CLASSIFICATION_DEFAULTS, "fraud")
        submitted_classification = st.form_submit_button("Classify claim")
    if submitted_classification:
        try:
            input_row = pd.DataFrame([classification_values])[CLASSIFICATION_FIELDS]
            probabilities = classification_pipeline.predict_proba(input_row)[0]
            class_labels = list(classification_pipeline.classes_)
            fraud_index = class_labels.index(1) if 1 in class_labels else 1
            fraud_probability = float(probabilities[fraud_index])
            is_fraud = fraud_probability >= classification_threshold
            st.success("Fraudulent" if is_fraud else "Legitimate")
            st.json({
                "fraud_probability": round(fraud_probability, 4),
                "decision_threshold": classification_threshold,
                "class_probabilities": {
                    "Legitimate": round(float(probabilities[1 - fraud_index]), 4),
                    "Fraudulent": round(fraud_probability, 4),
                },
            })
        except (TypeError, ValueError, KeyError) as error:
            st.error(str(error))

with tab_clustering:
    st.info("Clustering will be available after Amiir's bank segmentation is merged.")

with tab_recommender:
    st.subheader("Pharmacy basket recommendations")
    with st.form("pharmacy_form"):
        cart_text = st.text_input("Basket SKUs, comma-separated", "D05, D04")
        top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=3)
        submitted_recommender = st.form_submit_button("Get recommendations")
    if submitted_recommender:
        cart = {item.strip() for item in cart_text.split(",") if item.strip()}
        if not cart:
            st.error("Enter at least one SKU.")
        else:
            candidates = {}
            for antecedent, consequent_list in recommendation_lookup.items():
                if antecedent.issubset(cart):
                    for sku, confidence, lift in consequent_list:
                        if sku not in cart and (
                            sku not in candidates or confidence > candidates[sku][0]
                        ):
                            candidates[sku] = (confidence, lift)
            ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
            recommendations = [
                {
                    "sku": str(sku),
                    "product_name": sku_lookup.get(sku, str(sku)),
                    "confidence": round(float(confidence), 3),
                    "lift": round(float(lift), 2),
                }
                for sku, (confidence, lift) in ranked[:top_n]
            ]
            if recommendations:
                st.dataframe(recommendations, hide_index=True)
            else:
                st.warning("No rule matches this cart. Try other items.")