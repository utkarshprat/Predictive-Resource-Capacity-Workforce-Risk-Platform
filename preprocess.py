import pandas as pd


def load_and_preprocess(path):
    df = pd.read_csv(path)

    # 1. Clean Data
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Feature Engineering
    df['Available_Hours'] = (40 - df['OverTime'].map({'Yes': 8, 'No': 0}) - df['WorkLifeBalance'] * 2)
    df['Assigned_Hours'] = (30 + df['JobLevel'] * 3 + df['OverTime'].map({'Yes': 10, 'No': 0}))
    df['Task_Complexity'] = df['JobLevel'] * 2
    df['Deadline_Days'] = 30 - df['JobInvolvement'] * 3
    df['Skill'] = df['JobRole']
    df['Workload_Ratio'] = df['Assigned_Hours'] / df['Available_Hours'].clip(lower=1)

    # 3. BALANCED RISK LOGIC (For Realistic Feature Importance)
    # We want the AI to see that BOTH Workload and Deadlines matter.
    df['Historical_Risk'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Stress Logic (Combined factors)
    df['Risk'] = (
        (df['Workload_Ratio'] > 1.2) | 
        (df['Deadline_Days'] < 12) |
        ((df['Historical_Risk'] == 1) & (df['Workload_Ratio'] > 0.9))
    ).astype(int)

    # 4. Future Workload
    ot_effect = df['OverTime'].map({'Yes': 12, 'No': 4})
    df['Future_Workload'] = (df['Assigned_Hours'] + df['Task_Complexity'] * 1.5 + df['JobInvolvement'] * 2 + ot_effect + (df['YearsAtCompany'] * 0.5))

    return df