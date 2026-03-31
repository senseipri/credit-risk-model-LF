import os
import joblib
import numpy as np
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_data.joblib")

# ---------------- GLOBAL VARIABLES (LAZY LOAD) ----------------
model = None
scaler = None
features = None
cols_to_scale = None


# ---------------- LOAD ARTIFACTS ----------------
def load_artifacts():
    global model, scaler, features, cols_to_scale

    if model is None:
        logger.info("Loading model artifacts...")

        model_data = joblib.load(MODEL_PATH)

        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        cols_to_scale = model_data['cols_to_scale']

        logger.info("Model artifacts loaded successfully")


# ---------------- PREPARE INPUT ----------------
def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):

    logger.info("Preparing input data")

    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income and income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Dummy values (required for model compatibility)
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])

    # ---------------- SCALING ----------------
    try:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    except Exception as e:
        logger.error(f"Scaling failed: {str(e)}")
        raise

    # ---------------- FEATURE ALIGNMENT ----------------
    df = df.reindex(columns=features, fill_value=0)

    logger.info("Input prepared successfully")

    return df


# ---------------- MAIN PREDICT FUNCTION ----------------
def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):

    load_artifacts()

    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months,
        avg_dpd_per_delinquency, delinquency_ratio,
        credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    logger.info("Running model prediction")

    probability, credit_score, rating = calculate_credit_score(input_df)

    logger.info(f"Prediction done: prob={probability:.4f}, score={credit_score}, rating={rating}")

    top_features = get_top_risk_factors(input_df)

    return probability, credit_score, rating, top_features


# ---------------- CREDIT SCORE CALCULATION ----------------
def calculate_credit_score(input_df, base_score=300, scale_length=600):

    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability

    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating

def get_top_risk_factors(input_df, top_n=3):
    contributions = input_df.values[0] * model.coef_[0]

    feature_contrib = list(zip(features, contributions))

    # Sort by highest contribution (risk increasing)
    feature_contrib.sort(key=lambda x: x[1], reverse=True)

    top_features = feature_contrib[:top_n]

    return top_features