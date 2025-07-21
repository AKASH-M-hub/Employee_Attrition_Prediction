import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('attrition_rf_model.pkl')

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("👩‍💼 Employee Attrition Predictor")
st.markdown("Enter employee details below to predict if they might leave the company.")

# User input form
with st.form("attrition_form"):
    Age = st.slider("Age", 18, 60, 30)
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    MonthlyIncome = st.slider("Monthly Income", 1000, 20000, 5000)
    OverTime = st.selectbox("OverTime", ['Yes', 'No'])
    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 5)
    YearsAtCompany = st.slider("Years at Company", 0, 40, 3)
    
    submitted = st.form_submit_button("Predict Attrition")

# If form is submitted
if submitted:
    gender_encoded = 1 if Gender == 'Male' else 0
    overtime_encoded = 1 if OverTime == 'Yes' else 0
    satisfaction_score = JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction

    # Create a dataframe for prediction
    input_data = pd.DataFrame([{
        'Age': Age,
        'Gender': gender_encoded,
        'MonthlyIncome': MonthlyIncome,
        'OverTime': overtime_encoded,
        'TotalWorkingYears': TotalWorkingYears,
        'YearsAtCompany': YearsAtCompany,
        'SatisfactionScore': satisfaction_score
    }])

    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This employee is likely to stay. (Probability: {probability:.2f})")
