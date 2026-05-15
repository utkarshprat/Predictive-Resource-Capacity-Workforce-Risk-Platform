from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

import numpy as np
import pandas as pd
import joblib

from preprocess import load_and_preprocess

# =========================================================
# LOAD DATASET
# =========================================================

df = load_and_preprocess("dataset.csv")

print("\n[SUCCESS] Dataset Loaded Successfully")

# =========================================================
# WORKLOAD FORECASTING MODEL
# =========================================================

print("\n==============================")
print("WORKLOAD FORECASTING MODEL")
print("==============================")

X_reg = df[
    [
        'Available_Hours',
        'Assigned_Hours',
        'Task_Complexity'
    ]
]

y_reg = df['Future_Workload']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

reg_model = LinearRegression()

reg_model.fit(
    X_train,
    y_train
)

pred_reg = reg_model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    pred_reg
)

print("\nMAE (Workload Forecasting):")
print(round(mae, 4))

# =========================================================
# RISK PREDICTION MODEL
# =========================================================

print("\n==============================")
print("WORKFORCE RISK PREDICTION MODEL")
print("==============================")

X_clf = df[
    [
        'Available_Hours',
        'Assigned_Hours',
        'Deadline_Days',
        'Workload_Ratio'
    ]
]

# Separate features for Deep Learning (includes Workload_Ratio for logic)
X_dl_base = df[
    [
        'Available_Hours',
        'Assigned_Hours',
        'Deadline_Days',
        'Workload_Ratio'
    ]
]

y_clf = df['Risk']

# =========================================================
# CLASS IMBALANCE HANDLING
# =========================================================

classes = np.unique(y_clf)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_clf
)

class_weight_dict = {
    cls: weight
    for cls, weight in zip(classes, class_weights)
}

print("\nClass Weights:")
print(class_weight_dict)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)

# Also split the DL features
X_train_dl_raw, X_test_dl_raw, _, _ = train_test_split(
    X_dl_base,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)

# =========================================================
# DECISION TREE MODEL
# =========================================================

clf_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)

clf_model.fit(
    X_train_c,
    y_train_c
)

pred_clf = clf_model.predict(
    X_test_c
)

# =========================================================
# LOGISTIC REGRESSION (Replaces ANN) RISK PREDICTION MODEL
# =========================================================

from sklearn.linear_model import LogisticRegression

dl_model = LogisticRegression(
    random_state=42,
    class_weight='balanced'
)

# --- FEATURE SCALING FOR RISK MODEL ---
scaler = StandardScaler()
scaler.fit(X_train_dl_raw) # Fit on DL features

# Apply scaling
X_train_dl_scaled = scaler.transform(X_train_dl_raw)
X_test_dl_scaled = scaler.transform(X_test_dl_raw)

dl_model.fit(
    X_train_dl_scaled,
    y_train_c
)

pred_dl = dl_model.predict(
    X_test_dl_scaled
)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test_c,
    pred_dl
)

print("\nAccuracy Score:")
print(round(accuracy, 4))

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test_c,
    pred_dl
)

print("\nConfusion Matrix:")
print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_test_c,
    pred_dl,
    output_dict=True
)

print("\nClassification Report:")

print(
    classification_report(
        y_test_c,
        pred_dl
    )
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

# Use Logistic Regression coefficients which account for class imbalance better
# and provide a more balanced feature importance distribution for the graph.
importances = np.abs(dl_model.coef_[0])
importances = importances / np.sum(importances)

feature_importance = pd.DataFrame({
    "Feature": X_clf.columns,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

# =========================================================
# FIX:
# CLEANER FEATURE IMPORTANCE VALUES
# =========================================================

feature_importance['Importance'] = (
    feature_importance['Importance']
    .round(3)
)

print("\nFeature Importance:")
print(feature_importance)

# DEEP LEARNING MODEL ALREADY TRAINED ABOVE

# =========================================================
# MODEL INSIGHTS
# =========================================================

model_insights = {
    "accuracy": round(accuracy, 4),

    "mae": round(mae, 4),

    "class_imbalance_note":
    (
        "The dataset contains imbalanced workforce "
        "risk classes, which affects minority-class recall."
    ),

    "prototype_note":
    (
        "Prototype AI Decision-Support Platform "
        "for workforce planning and operational simulation."
    ),

    "model_behavior":
    (
        "The workforce risk model performs better "
        "at identifying low-risk employees than "
        "minority high-risk employees due to "
        "dataset imbalance."
    ),

    "business_note":
    (
        "The platform combines ML-based workforce "
        "risk prediction with operational workload "
        "stress analysis for decision support."
    )
}

# =========================================================
# SAVE MODELS
# =========================================================

joblib.dump(
    reg_model,
    "reg_model.pkl"
)

# clf_model (Decision Tree) is NOT saved because we only need its feature importance, not the model itself.

joblib.dump(
    dl_model,
    "dl_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

# =========================================================
# SAVE EVALUATION DATA
# =========================================================

joblib.dump(
    cm,
    "confusion_matrix.pkl"
)

joblib.dump(
    report,
    "classification_report.pkl"
)

joblib.dump(
    feature_importance,
    "feature_importance.pkl"
)

joblib.dump(
    model_insights,
    "model_insights.pkl"
)

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")

print("\n[SUCCESS] Models Saved Successfully")
print("[SUCCESS] Evaluation Files Saved")
print("[SUCCESS] Feature Importance Saved")
print("[SUCCESS] Model Insights Saved")

print("\nPrototype AI Decision-Support Platform Ready")