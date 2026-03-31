from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Tuple

from prediction_helper import predict
from logger import get_logger

# ---------------- LOGGER ----------------
logger = get_logger(__name__)

# ---------------- APP ----------------
app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML-powered API for predicting credit risk",
    version="1.0.0"
)

# ---------------- INPUT SCHEMA ----------------
class CreditRiskInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_tenure_months: int = Field(..., gt=0)
    avg_dpd_per_delinquency: float = Field(..., ge=0)
    delinquency_ratio: float = Field(..., ge=0)
    credit_utilization_ratio: float = Field(..., ge=0)
    num_open_accounts: int = Field(..., ge=0)
    residence_type: str
    loan_purpose: str
    loan_type: str


# ---------------- OUTPUT SCHEMA ----------------
class CreditRiskOutput(BaseModel):
    probability: float
    credit_score: int
    rating: str
    top_features: List[Tuple[str, float]]


# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health_check():
    logger.info("Health check endpoint hit")
    return {"status": "ok"}


# ---------------- PING ----------------
@app.get("/ping")
def ping():
    logger.info("Ping endpoint hit")
    return {"message": "Server is alive 🚀"}


# ---------------- PREDICTION ENDPOINT ----------------
@app.post("/predict_credit_risk", response_model=CreditRiskOutput)
def predict_credit_risk(input_data: CreditRiskInput):
    logger.info(f"Request received: {input_data.dict()}")

    try:
        # FIXED (4 values now)
        probability, credit_score, rating, top_features = predict(
            input_data.age,
            input_data.income,
            input_data.loan_amount,
            input_data.loan_tenure_months,
            input_data.avg_dpd_per_delinquency,
            input_data.delinquency_ratio,
            input_data.credit_utilization_ratio,
            input_data.num_open_accounts,
            input_data.residence_type,
            input_data.loan_purpose,
            input_data.loan_type
        )

        logger.info(
            f"Prediction successful | score={credit_score}, rating={rating}"
        )

        # RETURN UPDATED RESPONSE
        return CreditRiskOutput(
            probability=probability,
            credit_score=credit_score,
            rating=rating,
            top_features=top_features
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")