import pandas as pd

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# LOAD FINAL DATASET
# -----------------------------------

df = pd.read_csv("../data/final_diabetic_data.csv")

print("Dataset Loaded Successfully!")
print(df.shape)

# -----------------------------------
# FEATURES & TARGET
# -----------------------------------

X = df.drop('readmitted', axis=1)

y = df['readmitted']

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)
# -----------------------------------
# HANDLE CLASS IMBALANCE WITH SMOTE
# -----------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())
# -----------------------------------
# LOGISTIC REGRESSION
# -----------------------------------

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, lr_pred))

print("\nPrecision:")
print(precision_score(y_test, lr_pred))

print("\nRecall:")
print(recall_score(y_test, lr_pred))

print("\nF1 Score:")
print(f1_score(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# -----------------------------------
# RANDOM FOREST
# -----------------------------------

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nPrecision:")
print(precision_score(y_test, rf_pred))

print("\nRecall:")
print(recall_score(y_test, rf_pred))

print("\nF1 Score:")
print(f1_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# -----------------------------------
# XGBOOST
# -----------------------------------

print("\n==============================")
print("XGBOOST")
print("==============================")

xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train_smote, y_train_smote)

xgb_pred = xgb_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, xgb_pred))

print("\nPrecision:")
print(precision_score(y_test, xgb_pred))

print("\nRecall:")
print(recall_score(y_test, xgb_pred))

print("\nF1 Score:")
print(f1_score(y_test, xgb_pred))

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

import joblib

# Save trained model
joblib.dump(xgb_model, "../models/readmission_model.pkl")

print("\nModel saved successfully!")