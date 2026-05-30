import pandas as pd
import joblib

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("../models/readmission_model.pkl")

print("Model loaded successfully!")

# -----------------------------------
# LOAD FINAL DATASET
# -----------------------------------

df = pd.read_csv("../data/final_diabetic_data.csv")

# -----------------------------------
# TAKE ONE SAMPLE PATIENT
# -----------------------------------

sample_patient = df.drop('readmitted', axis=1).iloc[[0]]

# -----------------------------------
# PREDICT
# -----------------------------------

prediction = model.predict(sample_patient)

# -----------------------------------
# OUTPUT
# -----------------------------------

if prediction[0] == 1:
    print("\nHigh Risk of Readmission")
else:
    print("\nLow Risk of Readmission")