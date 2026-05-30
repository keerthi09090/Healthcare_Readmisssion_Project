import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../data/diabetic_data.csv")

# Dataset overview
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Replace missing values represented as '?'
df.replace('?', np.nan, inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False))

# Drop columns with too many missing values
drop_cols = ['weight', 'payer_code', 'medical_specialty']

df.drop(columns=drop_cols, inplace=True)

print("\nColumns after dropping:")
print(df.columns)

# Remove duplicate patients
df = df.drop_duplicates(subset='patient_nbr', keep='last')

print("\nShape after removing duplicate patients:")
print(df.shape)

# Remove unnecessary ID columns
df.drop(columns=['encounter_id', 'patient_nbr'], inplace=True)

# Convert target variable
df['readmitted'] = df['readmitted'].apply(
    lambda x: 1 if x == '<30' else 0
)

# Convert age ranges into numeric values
age_map = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}

df['age'] = df['age'].map(age_map)

# Final dataset info
print("\nFinal Dataset Shape:")
print(df.shape)

print("\nTarget Distribution:")
print(df['readmitted'].value_counts())

# Save cleaned dataset
df.to_csv("../data/cleaned_diabetic_data.csv", index=False)

print("\nCleaned dataset saved successfully!")