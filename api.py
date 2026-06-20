from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# =========================================================
# FASTAPI MICROSERVICE SETUP
# =========================================================

app = FastAPI(
    title="Workforce AI Microservice",
    description="Enterprise REST API for Workforce Risk & Capacity Predictions",
    version="1.0.0"
)

# =========================================================
# LOAD OUR AI MODELS
# =========================================================
try:
    reg_model = joblib.load("reg_model.pkl")  # The Linear Regression Model
    dl_model = joblib.load("dl_model.pkl")    # The Risk Prediction (Logistic) Model
    scaler = joblib.load("scaler.pkl")      # The Feature Scaler
    print("[SUCCESS] AI Models & Scaler Loaded Successfully into API")
except Exception as e:
    print(f"[ERROR] Error loading models: {e}")

# =========================================================
# DEFINE OUR INPUT DATA STRUCTURES (Pydantic Models)
# =========================================================


class WorkloadInput(BaseModel):
    available_hours: float
    assigned_hours: float
    task_complexity: float

class RiskInput(BaseModel):
    available_hours: float
    assigned_hours: float
    deadline_days: float

# =========================================================
# API ROUTE 1: PREDICT FUTURE WORKLOAD
# =========================================================

@app.post("/predict/workload")
def predict_workload(data: WorkloadInput):
    """
    Predicts the future workload of an employee using Linear Regression.
    """
    # 1. We extract the data sent to us into a format our AI understands
    features = [[
        data.available_hours,
        data.assigned_hours,
        data.task_complexity
    ]]
    
    # 2. We ask our Linear Regression model to predict the workload
    prediction = reg_model.predict(features)[0]
    
    # 3. We return a JSON response back to whoever called our API
    return {
        "status": "success",
        "model_used": "Linear Regression",
        "predicted_future_workload": round(prediction, 2)
    }

# =========================================================
# API ROUTE 2: PREDICT WORKFORCE RISK
# =========================================================

@app.post("/predict/risk")
def predict_risk(data: RiskInput):
    """
    Predicts the probability of an employee quitting using the Workforce Risk Prediction Model.
    """
    # 1. We extract the data sent to us and calculate the Workload Ratio
    workload_ratio = data.assigned_hours / max(1, data.available_hours)
    features = [[
        data.available_hours,
        data.assigned_hours,
        data.deadline_days,
        workload_ratio
    ]]
    
    # 2. We ask our model to predict the probability of Risk
    # We must SCALE the features first using the same scaler we used in training!
    features_scaled = scaler.transform(features)
    risk_probability = dl_model.predict_proba(features_scaled)[0][1]
    
    # 3. We return a JSON response
    return {
        "status": "success",
        "model_used": "Workforce Risk Prediction Model",
        "risk_percentage": round(risk_probability * 100, 2)
    }
