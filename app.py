# =========================================================
# Predictive Resource Capacity & Workforce Risk Platform
# PART 1
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from preprocess import load_and_preprocess

from utils import (
    decision_engine,
    explain_prediction,
    allocate_resources,
    suggest_upskilling
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Workforce Risk & Resource Capacity Platform",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# PREMIUM STYLING (Glassmorphism & Professional Theme)
# =========================================================

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stMetric:hover {
        border: 1px solid rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }

    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.02);
        padding: 2rem;
        border-radius: 20px;
    }

    h1 {
        color: #f5f5f5;
        font-weight: 700;
        font-size: 42px;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }

    p {
        color: #b0b3b8;
        font-size: 15px;
    }

    .stHeader {
        color: #dcdcdc;
    }

    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        border: none;
    }

    </style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

reg_model = joblib.load("reg_model.pkl")
dl_model = joblib.load("dl_model.pkl")
cm = joblib.load("confusion_matrix.pkl")
report = joblib.load("classification_report.pkl")
feature_importance = joblib.load("feature_importance.pkl")
model_insights = joblib.load("model_insights.pkl")
scaler = joblib.load("scaler.pkl")

# =========================================================
# TITLE
# =========================================================

st.title("🚀 Predictive Resource Capacity & Workforce Risk Platform")
st.caption("AI-powered workforce forecasting, operational stress analysis, workforce risk prediction, and intelligent resource optimization platform")

required_columns = [
    'OverTime', 'WorkLifeBalance', 'JobLevel', 'JobInvolvement',
    'JobRole', 'Attrition', 'MonthlyIncome', 'YearsAtCompany'
]

# =========================================================
# DATASET UPLOAD
# =========================================================

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

try:
    if uploaded_file:
        temp_df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in required_columns if col not in temp_df.columns]
        if missing_cols:
            st.error(f"Missing Required Columns: {missing_cols}")
            st.stop()
        uploaded_file.seek(0)
        df = load_and_preprocess(uploaded_file)
    else:
        df = load_and_preprocess("dataset.csv")
except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# =========================================================
# INITIAL ORGANIZATION METRICS
# =========================================================

st.header("🏢 Organization Overview")
initial_stress = ((df['Assigned_Hours'] > (df['Available_Hours'] * 1.15))).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(df))
col2.metric("Employees Under Stress", int(initial_stress))
col3.metric("Average Workload", round(df['Assigned_Hours'].mean(), 2))

# =========================================================
# UI CONTAINERS (Define Layout Order)
# =========================================================

ui_emp_intel_header = st.container()
ui_emp_details = st.container()
ui_emp_intel_scenario = st.container()
ui_emp_results = st.container()

ui_proj_intel_inputs = st.container()
ui_proj_results = st.container()

ui_org_intel = st.container()
ui_eval = st.container()

# =========================================================
# INPUT PHASE (Collects all Sliders/Selects)
# =========================================================

with ui_emp_intel_header:
    st.header("👤 Employee Intelligence")
    employee_index = st.selectbox("Select Employee", df.index)
    selected_employee = df.loc[employee_index]
    employee_skill = selected_employee['Skill']

    base_available = int(selected_employee['Available_Hours'])
    base_assigned = int(selected_employee['Assigned_Hours'])
    base_deadline = int(selected_employee['Deadline_Days'])
    base_complexity = int(selected_employee['Task_Complexity'])

st.sidebar.header("⚙️ Employee Simulation")
available_adjust = st.sidebar.slider("Available Hours Adjustment", -20, 20, 0)
assigned_adjust = st.sidebar.slider("Assigned Hours Adjustment", -20, 30, 0)
complexity_adjust = st.sidebar.slider("Task Complexity Adjustment", -5, 5, 0)
deadline_adjust = st.sidebar.slider("Deadline Days Adjustment", -15, 15, 0)

with ui_emp_intel_scenario:
    scenario = st.selectbox(
        "Select Scenario",
        ["Normal", "Increase Workload", "Employee Leaves", "New Project"]
    )

with ui_proj_intel_inputs:
    st.header("🧩 Project Intelligence")
    required_skill = st.selectbox("Select Required Skill", sorted(df['Skill'].unique()))
    project_a = st.number_input("Project A Workforce Requirement", 0, 50, 5)
    project_b = st.number_input("Project B Workforce Requirement", 0, 50, 3)
    project_complexity = st.slider("Project Complexity", 1, 10, 5)
    project_deadline_pressure = st.slider("Project Deadline Pressure", 1, 30, 10)

# =========================================================
# SIMULATION & MATH PHASE (Unified Data Pipeline)
# =========================================================

# 1. Base Copy
org_sim = df.copy()

# 2. Employee Sliders (Individual Adjustments)
org_sim.loc[employee_index, 'Available_Hours'] = max(1, base_available + available_adjust)
org_sim.loc[employee_index, 'Assigned_Hours'] = max(1, base_assigned + assigned_adjust)
org_sim.loc[employee_index, 'Task_Complexity'] = max(1, base_complexity + complexity_adjust)
org_sim.loc[employee_index, 'Deadline_Days'] = max(1, base_deadline + deadline_adjust)

# 3. Ripple Effect (Redistribution to the REST of the team, ignoring the selected employee)
same_skill_mask = (org_sim['Skill'] == employee_skill)
ripple_mask = same_skill_mask & (org_sim.index != employee_index)

# Round the ripple effect to keep the graph's visual columns clean
ripple_impact = round(assigned_adjust * 0.05)
org_sim.loc[ripple_mask, 'Assigned_Hours'] -= ripple_impact

# 4. Project Effect (Applies to everyone with required_skill, including selected employee)
project_total_demand = project_a + project_b
available_skill_count = len(df[df['Skill'] == required_skill])
project_skill_gap = max(0, project_total_demand - available_skill_count)

# Round the project load to keep the graph's visual columns clean
organization_project_load = round(project_total_demand * 0.08)
project_team_mask = (org_sim['Skill'] == required_skill)
org_sim.loc[project_team_mask, 'Assigned_Hours'] += organization_project_load

if organization_project_load > 0:
    project_deadline_impact = round(project_deadline_pressure * 0.15)
    org_sim.loc[project_team_mask, 'Deadline_Days'] -= project_deadline_impact

# 5. Scenario Effect (Applies to everyone, unified impact)
if scenario == "Increase Workload":
    org_sim['Assigned_Hours'] += 3
    org_sim.loc[employee_index, 'Assigned_Hours'] += 2  # additional impact for selected employee
    org_sim.loc[employee_index, 'Task_Complexity'] += 1
elif scenario == "Employee Leaves":
    org_sim['Assigned_Hours'] += 6
    org_sim.loc[employee_index, 'Assigned_Hours'] += 2
    org_sim.loc[employee_index, 'Task_Complexity'] += 2
elif scenario == "New Project":
    org_sim['Assigned_Hours'] += 9
    org_sim.loc[employee_index, 'Assigned_Hours'] += 3
    org_sim.loc[employee_index, 'Task_Complexity'] += 3

# 6. Cleanup limits
org_sim['Assigned_Hours'] = org_sim['Assigned_Hours'].clip(lower=1)
org_sim['Available_Hours'] = org_sim['Available_Hours'].clip(lower=1)
org_sim['Deadline_Days'] = org_sim['Deadline_Days'].clip(lower=1)
org_sim['Task_Complexity'] = org_sim['Task_Complexity'].clip(lower=1)

# 6.5 New Feature: Workload Ratio
org_sim['Workload_Ratio'] = org_sim['Assigned_Hours'] / org_sim['Available_Hours']

# 7. ML Predictions for Organization
org_sim_features = org_sim[['Available_Hours', 'Assigned_Hours', 'Deadline_Days', 'Workload_Ratio']]
org_sim_scaled = scaler.transform(org_sim_features)

org_sim['Simulated_Risk'] = dl_model.predict_proba(
    org_sim_scaled
)[:, 1]

org_sim['Operational_Stress'] = (
    (org_sim['Assigned_Hours'] > (org_sim['Available_Hours'] * 1.15)) |
    (org_sim['Simulated_Risk'] > 0.50)
).astype(int)

org_sim['Risk_Zone'] = np.where(org_sim['Operational_Stress'] == 1, 'High Stress', 'Low Stress')

# 8. Extract Final Employee State for UI
employee_final_state = org_sim.loc[employee_index]
employee_available = employee_final_state['Available_Hours']
employee_assigned = employee_final_state['Assigned_Hours']
employee_deadline = employee_final_state['Deadline_Days']
employee_complexity = employee_final_state['Task_Complexity']
employee_risk = employee_final_state['Simulated_Risk']
employee_stress = employee_final_state['Operational_Stress'] == 1

employee_future_workload = reg_model.predict([[
    employee_available,
    employee_assigned,
    employee_complexity
]])[0]

# 9. Project Math
availability_ratio = project_total_demand / max(1, available_skill_count)
project_risk = min(1.0, 
    (availability_ratio * 0.50) + 
    ((project_complexity / 10) * 0.30) + 
    (project_deadline_pressure / 30) * 0.20
)
project_future_workload = project_total_demand * project_complexity

# 10. Org Math
org_future_workload = reg_model.predict(
    org_sim[['Available_Hours', 'Assigned_Hours', 'Task_Complexity']]
).mean()
org_risk = org_sim['Simulated_Risk'].mean()
org_stress_count = org_sim['Operational_Stress'].sum()

# =========================================================
# RENDERING PHASE (Inject UI into Containers)
# =========================================================

with ui_emp_details:
    employee_details = pd.DataFrame({
        "Feature": ["Skill", "Available Hours", "Assigned Hours", "Deadline Days", "Task Complexity"],
        "Value": [str(employee_skill), str(base_available), str(base_assigned), str(base_deadline), str(base_complexity)]
    })
    st.dataframe(employee_details, use_container_width=True)

with ui_emp_results:
    st.subheader("📊 Employee Prediction Results")
    col_emp1, col_emp2, col_emp3 = st.columns(3)
    col_emp1.metric("Future Workload", round(employee_future_workload, 2))
    col_emp2.metric("Employee Risk", f"{employee_risk * 100:.2f}%")
    col_emp3.metric("Stress Status", "High" if employee_stress else "Normal")

    st.subheader("💡 Employee Recommendation")
    employee_recommendation = decision_engine(employee_available, employee_assigned, employee_risk, 0)
    st.info(employee_recommendation)

    st.subheader("🔍 Employee Prediction Explanation")
    employee_explanation = explain_prediction(employee_available, employee_assigned, employee_deadline, employee_risk)
    st.write(employee_explanation)

    st.subheader("📈 Employee Workload Analysis")
    fig_emp = plt.figure(figsize=(5, 3))
    employee_graph_df = pd.DataFrame({
        "Metric": ["Available Hours", "Assigned Hours"],
        "Value": [employee_available, employee_assigned]
    })
    sns.barplot(data=employee_graph_df, x='Metric', y='Value')
    plt.ylabel("Hours")
    st.pyplot(fig_emp, use_container_width=False)

with ui_proj_results:
    st.subheader("📊 Project Prediction Results")
    col_proj1, col_proj2, col_proj3, col_proj4 = st.columns(4)
    col_proj1.metric("Project Workforce Demand", project_total_demand)
    col_proj2.metric("Project Future Workload", project_future_workload)
    col_proj3.metric("Project Risk", f"{project_risk * 100:.2f}%")
    col_proj4.metric("Project Skill Gap", project_skill_gap)

    if project_skill_gap > 0:
        st.error(f"⚠️ Need {project_skill_gap} additional employees")
    else:
        st.success("✅ Workforce capacity sufficient")

    st.subheader("📈 Project Workforce Analysis")
    fig_proj = plt.figure(figsize=(5, 3))
    project_graph_df = pd.DataFrame({
        "Category": ["Required Workforce", "Available Workforce"],
        "Value": [project_total_demand, available_skill_count]
    })
    sns.barplot(data=project_graph_df, x='Category', y='Value')
    plt.ylabel("Employees")
    st.pyplot(fig_proj, use_container_width=False)

with ui_org_intel:
    st.header("🏢 Organization Intelligence")
    col_org1, col_org2, col_org3 = st.columns(3)
    col_org1.metric("Organization Workload", round(org_future_workload, 2))
    col_org2.metric("Organization Risk", f"{org_risk * 100:.2f}%")
    col_org3.metric("Employees Under Stress", int(org_stress_count))

    st.subheader("🔥 Organization Risk Heatmap")
    fig_heat = plt.figure(figsize=(6, 3.5))
    sns.scatterplot(data=org_sim, x='Assigned_Hours', y='Simulated_Risk', hue='Risk_Zone', alpha=0.7)
    
    final_employee_row = org_sim.loc[employee_index]
    plt.scatter(
        final_employee_row['Assigned_Hours'],
        final_employee_row['Simulated_Risk'],
        s=250, marker='X', label='Selected Employee'
    )
    plt.xlabel("Assigned Hours")
    plt.ylabel("Risk Probability")
    plt.legend()
    st.pyplot(fig_heat, use_container_width=False)

    st.subheader("📊 Organization Stress Distribution")
    fig_stress = plt.figure(figsize=(5, 3))
    org_sim['Risk_Zone'].value_counts().reindex(['Low Stress', 'High Stress'], fill_value=0).plot.bar()
    plt.xlabel("Stress Level")
    plt.ylabel("Employees")
    st.pyplot(fig_stress, use_container_width=False)

    st.subheader("🚨 Top Risk Employees")
    top_risk = org_sim.sort_values(by='Simulated_Risk', ascending=False).head(5)
    st.dataframe(top_risk[['Skill', 'Assigned_Hours', 'Available_Hours', 'Simulated_Risk']], use_container_width=True)

    st.subheader("⚠️ Top Overloaded Employees")
    overload_df = org_sim.copy()
    overload_df['Overload'] = overload_df['Assigned_Hours'] - overload_df['Available_Hours']
    top_overload = overload_df.sort_values(by='Overload', ascending=False).head(5)
    st.dataframe(top_overload[['Skill', 'Assigned_Hours', 'Available_Hours', 'Overload']], use_container_width=True)

    st.subheader("🎓 Upskilling Recommendation")
    upskill = suggest_upskilling(org_sim, required_skill, project_skill_gap)
    if isinstance(upskill, list):
        for item in upskill:
            st.write(f"• {item}")
    else:
        if "🚨" in upskill:
            st.error(upskill)
        else:
            st.success(upskill)

    st.subheader("⚙️ Resource Optimization")
    allocation = allocate_resources(org_sim, required_skill, project_total_demand)
    st.dataframe(allocation, use_container_width=True)

    st.subheader("📈 Workforce Trend Analytics")
    trend_df = pd.DataFrame({
        "Metric": [
            "Average Assigned Hours",
            "Average Available Hours",
            "Average Risk Probability",
            "Employees Under Stress"
        ],
        "Value": [
            round(org_sim['Assigned_Hours'].mean(), 2),
            round(org_sim['Available_Hours'].mean(), 2),
            round(org_sim['Simulated_Risk'].mean(), 2),
            int(org_sim['Operational_Stress'].sum())
        ]
    })
    st.dataframe(trend_df, use_container_width=True)

with ui_eval:
    st.header("🧠 Model Evaluation")
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.subheader("Confusion Matrix")
        fig_cm = plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig_cm, use_container_width=False)
    
    with col_eval2:
        st.subheader("Classification Metrics")
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        st.metric("Precision", f"{precision:.2f}")
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1 Score", f"{f1:.2f}")

    st.subheader("📌 Feature Importance")
    fig_imp = plt.figure(figsize=(5, 3))
    sns.barplot(data=feature_importance, x='Feature', y='Importance')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_imp, use_container_width=False)

    st.subheader("📝 Model Insights")
    st.warning(model_insights['class_imbalance_note'])
    st.info(model_insights['prototype_note'])

#     st.header("🏗️ System Architecture")
#     st.markdown("""
# ### Workflow Pipeline
# 1. HR Dataset Input
# 2. Data Preprocessing
# 3. Feature Engineering
# 4. Employee Intelligence
# 5. Project Intelligence
# 6. Organization Intelligence
# 7. ML Prediction Models
# 8. Workforce Risk Analysis
# 9. Resource Optimization
# 10. Analytics Dashboard

# ---

# ### AI Components & Architecture
# - Machine Learning (Linear Regression) → Future Workload Forecasting
# - ML Classification (Logistic Regression) → Workforce Risk Prediction (High Precision)
# - Explainable AI (Decision Tree) → Model Interpretability (XAI)
# - FastAPI REST Microservice → Enterprise Backend Integration
# - Decision Engine → AI Recommendations
# - Optimization Engine → Resource Allocation

# ---

# ### Business Goals
# - Workforce forecasting
# - Workforce optimization
# - Operational stress detection
# - Workforce risk analysis
# - Project staffing analysis
# - Resource capacity planning
# - Skill gap analysis
# """)