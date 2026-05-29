# README.md

# AI-Powered Diabetic Patient Readmission Prediction System

## Overview

This project is an end-to-end Healthcare Machine Learning application designed to predict whether a diabetic patient is at risk of hospital readmission.

Hospital readmissions are a major challenge in healthcare systems, increasing operational costs and impacting patient outcomes. This project uses machine learning techniques to analyze patient records and predict readmission risk.

The application includes:

* Data cleaning pipeline
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine learning model training
* Class imbalance handling using SMOTE
* XGBoost prediction model
* Streamlit web application for real-time predictions

---

# Business Problem

Hospital readmissions are costly and often preventable. Early prediction of high-risk patients helps hospitals:

* Improve patient care
* Reduce healthcare costs
* Optimize hospital resources
* Enable proactive interventions

This project predicts whether a diabetic patient will be readmitted within 30 days after discharge.

---

# Dataset

Dataset Used:

* Diabetes 130-US Hospitals Dataset

Dataset contains:

* 100,000+ patient records
* 50+ healthcare-related features
* Patient demographics
* Medication details
* Lab procedures
* Hospital visits
* Readmission status

---

# Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Joblib
* Streamlit

---

# Machine Learning Workflow

## 1. Data Cleaning

* Handled missing values
* Removed duplicate patient records
* Removed unnecessary columns
* Converted target variable into binary classification

## 2. Exploratory Data Analysis

* Readmission distribution
* Age distribution
* Medication analysis
* Correlation heatmaps
* Hospital stay analysis

## 3. Feature Engineering

* Label encoding
* Categorical variable transformation
* ML-ready dataset creation

## 4. Model Training

Models used:

* Logistic Regression
* Random Forest
* XGBoost

## 5. Imbalance Handling

Implemented:

* SMOTE (Synthetic Minority Oversampling Technique)

## 6. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

# Streamlit Web Application

The project includes a Streamlit-based web application that allows users to:

* Enter patient information
* Predict readmission risk in real-time
* View prediction results instantly

---

# Project Structure

```bash
Healthcare_Readmission_Project/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│   └── readmission_model.pkl
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# How To Run The Project

## 1. Clone Repository

```bash
git clone <repository-link>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Streamlit App

```bash
python3 -m streamlit run app/app.py
```

---

# Results

The machine learning pipeline successfully predicts diabetic patient readmission risk using healthcare data.

Implemented advanced ML techniques such as:

* SMOTE
* XGBoost
* Feature engineering
* Model persistence

---

# Future Improvements

* Hyperparameter tuning
* Better recall optimization
* Feature importance visualization
* Cloud deployment
* Real-time hospital integration

---

# Author

Keerthi
