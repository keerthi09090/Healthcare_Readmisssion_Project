import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Healthcare Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = joblib.load("models/readmission_model.pkl")

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("Diabetic Patient Readmission Prediction")
st.markdown("---")


st.write("Predict whether a diabetic patient is at risk of hospital readmission.")
col1, col2, col3 = st.columns(3)

col1.metric("Model", "XGBoost")

col2.metric("Dataset", "100K+")

col3.metric("Recall Focus", "Healthcare AI")

# -----------------------------------
# USER INPUTS
# -----------------------------------

age = st.slider("Age", 0, 100, 50)

time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)

num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 40)

num_medications = st.slider("Number of Medications", 1, 50, 10)

number_emergency = st.slider("Emergency Visits", 0, 20, 0)

number_inpatient = st.slider("Inpatient Visits", 0, 20, 0)

# -----------------------------------
# PREDICT BUTTON
# -----------------------------------

if st.button("Predict Readmission Risk"):

    # Load ML-ready dataset
    full_df = pd.read_csv("data/final_diabetic_data.csv")

    # Take one sample row
    input_df = full_df.drop("readmitted", axis=1).iloc[[0]].copy()

    # Replace values from UI
    input_df["age"] = age
    input_df["time_in_hospital"] = time_in_hospital
    input_df["num_lab_procedures"] = num_lab_procedures
    input_df["num_medications"] = num_medications
    input_df["number_emergency"] = number_emergency
    input_df["number_inpatient"] = number_inpatient

    # Predict
    prediction = model.predict(input_df)

    prediction_proba = model.predict_proba(input_df)[0][1]

    risk_percent = round(prediction_proba * 100, 2)

    # Output
    if prediction[0] == 1:
        st.error(
            f"⚠️ High Risk of Readmission ({risk_percent}% probability)"
        )
    else:
        st.success(
            f"✅ Low Risk of Readmission ({risk_percent}% probability)"
        )

st.markdown("---")

st.caption(
    "Built using Python, XGBoost, SMOTE, and Streamlit"
)