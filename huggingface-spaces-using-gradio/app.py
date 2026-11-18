import os

import gradio as gr
import joblib
import pandas as pd

# -----------------------------------------------------------------------
# Load the same saved artifacts as the API. Hugging Face receives a model/
# copy; local runs can use the repository's root model/ folder.
# -----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
if not os.path.isdir(MODEL_DIR):
    MODEL_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "model"))


def load_artifact(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


regression_pipeline = load_artifact("student_exam_score_regressor.joblib")
fraud_artifact = load_artifact("fraud_model_bundle.joblib")
classification_pipeline = fraud_artifact["model"]
classification_threshold = float(fraud_artifact["threshold"])
arm_artifact = load_artifact("arm_pharmacy_basket_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]
REGRESSION_FIELDS = list(regression_pipeline.feature_names_in_)
CLASSIFICATION_FIELDS = list(fraud_artifact["features"])

REGRESSION_DEFAULTS = {
    "major_subject": "Engineering", "parental_education_level": "Bachelor's",
    "internet_access_quality": "Good", "part_time_job": "No",
    "study_group_participation": "Yes", "study_hours_per_week": 9.5,
    "attendance_rate_pct": 88.0, "previous_gpa": 3.4,
    "sleep_hours_per_night": 7.0, "extracurricular_activities_count": 1,
    "stress_level_score": 5.0, "tutoring_hours_per_week": 2.0,
}
CLASSIFICATION_DEFAULTS = {
    "policy_type": "Comprehensive", "claim_type": "Collision",
    "incident_location_type": "Highway", "witnesses_present": "No",
    "agent_flagged_history": "Yes", "police_report_filed": "No",
    "claimant_age": 34, "policy_tenure_years": 1.2,
    "prior_claims_count": 3, "days_to_report_claim": 18,
    "claim_amount_kes": 480000.0,
    "claim_amount_to_policy_limit_ratio": 0.92, "customer_credit_score": 540.0,
}


def make_inputs(fields, defaults):
    return [
        gr.Textbox(label=field, value=defaults[field])
        if isinstance(defaults[field], str)
        else gr.Number(label=field, value=defaults[field])
        for field in fields
    ]


# -----------------------------------------------------------------------
# 1. Regression -- predict a student's final exam score
# -----------------------------------------------------------------------
def predict_exam_score(*values):
    row = pd.DataFrame([dict(zip(REGRESSION_FIELDS, values))])
    prediction = float(regression_pipeline.predict(row)[0])
    return {
        "model": "student_exam_score_regressor",
        "predicted_final_exam_score": round(prediction, 2),
    }


# -----------------------------------------------------------------------
# 2. Classification -- predict insurance claim fraud
# -----------------------------------------------------------------------
def predict_fraud(*values):
    row = pd.DataFrame([dict(zip(CLASSIFICATION_FIELDS, values))])
    probabilities = classification_pipeline.predict_proba(row)[0]
    class_labels = list(classification_pipeline.classes_)
    fraud_index = class_labels.index(1) if 1 in class_labels else 1
    fraud_probability = float(probabilities[fraud_index])
    legit_probability = float(probabilities[1 - fraud_index])
    is_fraud = fraud_probability >= classification_threshold
    return {
        "model": fraud_artifact.get("model_name", "Insurance fraud classifier"),
        "predicted_claim_status": "Fraudulent" if is_fraud else "Legitimate",
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_probability, 4),
        "decision_threshold": classification_threshold,
        "class_probabilities": {
            "Legitimate": round(legit_probability, 4),
            "Fraudulent": round(fraud_probability, 4),
        },
    }


# -----------------------------------------------------------------------
# 3. Clustering -- pending Member 3's merge to main
# -----------------------------------------------------------------------
def predict_segment(*_values):
    raise gr.Error("Clustering will be available after Amiir's bank model is merged.")


# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend products for a shopping cart
# -----------------------------------------------------------------------
def recommend_products(cart_text, top_n):
    cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
    if not cart:
        raise gr.Error("Enter at least one SKU, separated by commas.")
    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
    return {
        "model": "pharmacy association rules",
        "cart": sorted(cart),
        "recommendations": [
            {
                "sku": str(sku),
                "product_name": sku_lookup.get(sku, str(sku)),
                "confidence": round(float(confidence), 3),
                "lift": round(float(lift), 2),
            }
            for sku, (confidence, lift) in ranked[:int(top_n)]
        ],
        "message": "" if ranked else "No rule matches this cart. Try other items.",
    }


# -----------------------------------------------------------------------
# One Gradio app with the three merged models and the pending clustering tab.
# -----------------------------------------------------------------------
with gr.Blocks(title="BBT 4206 Model Demos") as demo:
    gr.Markdown("# BBT 4206 Model Demos")

    with gr.Tab("Exam score"):
        gr.Interface(
            fn=predict_exam_score,
            inputs=make_inputs(REGRESSION_FIELDS, REGRESSION_DEFAULTS),
            outputs=gr.JSON(label="Prediction"),
        )

    with gr.Tab("Fraud classification"):
        gr.Interface(
            fn=predict_fraud,
            inputs=make_inputs(CLASSIFICATION_FIELDS, CLASSIFICATION_DEFAULTS),
            outputs=gr.JSON(label="Prediction"),
        )

    with gr.Tab("Clustering · pending"):
        gr.Markdown("Clustering will be available after Amiir's bank segmentation is merged.")

    with gr.Tab("Pharmacy recommender"):
        gr.Interface(
            fn=recommend_products,
            inputs=[
                gr.Textbox(label="Basket SKUs", value="D05, D04"),
                gr.Slider(1, 10, value=3, step=1, label="Recommendations"),
            ],
            outputs=gr.JSON(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()