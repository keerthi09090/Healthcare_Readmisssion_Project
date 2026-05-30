import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_diabetic_data.csv")

print("Dataset Shape:")
print(df.shape)

# -----------------------------------
# CHECK CATEGORICAL COLUMNS
# -----------------------------------

categorical_cols = df.select_dtypes(include='object').columns

print("\nCategorical Columns:")
print(categorical_cols)

# -----------------------------------
# REMOVE INVALID GENDER VALUES
# -----------------------------------

df = df[df['gender'] != 'Unknown/Invalid']

# -----------------------------------
# LABEL ENCODING
# -----------------------------------

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("\nEncoding Complete!")

# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------

X = df.drop('readmitted', axis=1)

y = df['readmitted']

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# -----------------------------------
# SAVE FINAL DATASET
# -----------------------------------

final_df = pd.concat([X, y], axis=1)

final_df.to_csv("../data/final_diabetic_data.csv", index=False)

print("\nFinal ML-ready dataset saved successfully!")