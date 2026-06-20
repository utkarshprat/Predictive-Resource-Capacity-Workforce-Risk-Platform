# Predictive Resource Capacity & Workforce Risk Platform

An AI-powered decision-support system that forecasts workforce capacity, detects operational stress, predicts workforce flight risk, and provides resource optimization recommendations.

---

## 🏗️ Project Architecture
The platform is split into two main components:
1. **FastAPI Microservice Backend (`api.py`)**: An enterprise-ready REST API that hosts the machine learning prediction models (Linear Regression for workload forecasting and Logistic Regression for risk prediction).
2. **Streamlit Frontend Dashboard (`app.py`)**: A user-friendly, interactive analytical web interface for HR managers, featuring employee intelligence simulation, what-if scenarios, and project risk diagnostics.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/utkarshprat/Predictive-Resource-Capacity-Workforce-Risk-Platform.git
cd Predictive-Resource-Capacity-Workforce-Risk-Platform
```

### 3. Create a Virtual Environment & Install Dependencies
```powershell
# Create environment
python -m venv venv

# Activate environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Running the Platform

### Step 1: Start the FastAPI Backend
Launch the API server using Uvicorn:
```bash
uvicorn api:app --reload
```
* **API URL**: http://127.0.0.1:8000
* **Interactive Documentation (Swagger UI)**: http://127.0.0.1:8000/docs

### Step 2: Start the Streamlit Frontend
Launch the analytical dashboard:
```bash
streamlit run app.py
```
* **Dashboard URL**: http://localhost:8501
