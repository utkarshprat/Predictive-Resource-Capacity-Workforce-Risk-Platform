# ---------------- SMART DECISION ENGINE ----------------

def decision_engine(
    available,
    assigned,
    risk_score,
    skill_gap
):

    if risk_score > 0.6 and assigned > available:
        return "🚨 Critical: Immediate hiring required"

    elif risk_score > 0.5:
        return "⚠️ Moderate Risk: Monitor workload and deadlines"

    elif skill_gap > 5:
        return f"⚠️ Significant Skill Gap: Hire {skill_gap} employees"

    elif assigned > available:
        return "⚠️ Overloaded: Reallocate tasks"

    elif assigned < available * 0.6:
        return "ℹ️ Underutilized: Assign more projects"

    else:
        return "✅ Balanced system"


# ---------------- SMART RESOURCE OPTIMIZATION ----------------

def allocate_resources(
    df,
    required_skill,
    required_count
):

    candidates = df[
        df['Skill'] == required_skill
    ].copy()

    # Lower workload = better
    # Higher availability = better
    # Lower risk = better

    risk_col = 'Simulated_Risk' if 'Simulated_Risk' in candidates.columns else 'Risk'
    
    candidates['Optimization_Score'] = (
        candidates['Available_Hours']
        - candidates['Assigned_Hours']
        - (candidates[risk_col] * 10)
    )

    candidates = candidates.sort_values(
        by='Optimization_Score',
        ascending=False
    )

    allocated = candidates.head(required_count)

    return allocated[[
        'Skill',
        'Assigned_Hours',
        'Available_Hours',
        'Optimization_Score'
    ]]


# ---------------- EXPLANATION ENGINE ----------------

def explain_prediction(
    available,
    assigned,
    deadline,
    risk_score
):

    reasons = []

    if assigned > available:
        reasons.append(
            "Workload exceeds employee capacity"
        )

    if deadline < 5:
        reasons.append(
            "Very tight deadline detected"
        )

    if risk_score > 0.7:
        reasons.append(
            "High predicted workforce instability"
        )

    if not reasons:
        return "System stable with balanced workload"

    return " | ".join(reasons)


# ---------------- UPSKILLING RECOMMENDATION ----------------

def suggest_upskilling(
    df,
    required_skill,
    skill_gap
):

    if skill_gap <= 0:
        return "No upskilling required"

    # Find employees with low workload who don't already have the required skill
    low_workload = df[df['Skill'] != required_skill].sort_values(
        by='Assigned_Hours'
    ).head(skill_gap)

    suggestions = []

    for idx, row in low_workload.iterrows():

        suggestions.append(
            f"Train employee in '{row['Skill']}' → {required_skill}"
        )

    if not suggestions:
        return f"🚨 Cannot upskill: Need {skill_gap} employees, but no cross-training candidates available."

    return suggestions