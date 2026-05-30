import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_diabetic_data.csv")

# Basic info
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# -------------------------------
# TARGET VARIABLE DISTRIBUTION
# -------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='readmitted', data=df)

plt.title("Readmission Distribution")
plt.xlabel("Readmitted")
plt.ylabel("Count")

plt.show()

# -------------------------------
# AGE DISTRIBUTION
# -------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df['age'], bins=10)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# -------------------------------
# GENDER DISTRIBUTION
# -------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='gender', data=df)

plt.title("Gender Distribution")

plt.show()

# -------------------------------
# TIME IN HOSPITAL
# -------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(x='readmitted', y='time_in_hospital', data=df)

plt.title("Hospital Stay vs Readmission")

plt.show()

# -------------------------------
# NUMBER OF MEDICATIONS
# -------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df['num_medications'], bins=20)

plt.title("Medication Distribution")

plt.show()

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,10))

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()