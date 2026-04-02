# AI-Enabled-Visa-Status-Prediction.
The project I am working on currently which is AI Enabled Visa Status Prediction and Processing Time Estimator. This is the part of my Infosys Internship 6.0 .

# 🌍 AI Enabled Visa Status Prediction & Processing Time Estimator

An end-to-end Machine Learning system to predict:
- ✅ Visa Case Status (Certified / Denied)
- ⏱️ Visa Processing Time (in days)

Developed as part of **Infosys Internship 6.0**

---

# 📌 Project Overview

This project leverages historical visa disclosure data to:
- Predict processing duration using regression models
- Classify visa outcomes using machine learning
- Provide an interactive web-based interface for users

---

# 🎯 Objectives

- Improve transparency in visa processing timelines  
- Assist applicants in estimating approval chances  
- Build a real-world ML pipeline from data to deployment  

---

# 🧩 Milestone 1: Data Collection & Preprocessing

## ✅ Objectives
- Build structured dataset from raw visa disclosure data
- Clean and preprocess data for modeling

## 🔧 Tasks Completed
- Loaded dataset (`LCA_Disclosure_Data_FY2026_Q1.xlsx`)
- Handled missing values and invalid entries
- Converted date columns (`RECEIVED_DATE`, `DECISION_DATE`)
- Created target variable:
  - `processing_days = DECISION_DATE - RECEIVED_DATE`
- Encoded categorical variables:
  - VISA_CLASS
  - EMPLOYER_COUNTRY
  - WORKSITE_STATE
- Saved cleaned dataset

## 📂 Output Files
- `data/processed/visa_processed.csv`
- `sample_input.csv`

---

# 📊 Milestone 2: Exploratory Data Analysis (EDA)

## ✅ Objectives
- Understand patterns in visa processing
- Identify trends and feature importance

## 🔧 Tasks Completed
- Visualized processing time distribution
- Analyzed trends across:
  - Visa types
  - Countries
  - Worksite states
- Identified seasonal trends (peak months)
- Generated correlation heatmap
- Created summary statistics

## 📂 Output Files
- `reports/*.png` (EDA charts)
- Insights for feature engineering

---

# 🤖 Milestone 3: Predictive Modeling

## ✅ Objectives
- Train regression and classification models
- Evaluate model performance

## 🔧 Regression Models
- Linear Regression
- Random Forest
- Gradient Boosting

## 🔧 Classification Models
- Logistic Regression
- Random Forest (Balanced)
- Gradient Boosting

## ⚙️ Feature Engineering
- APPLICATION_MONTH
- APPLICATION_YEAR
- IS_PEAK_SEASON
- TOTAL_EMPLOYMENT_CHANGES
- WAGE_DIFF

## 📈 Evaluation Metrics
- Regression:
  - MAE
  - RMSE
  - R² Score
- Classification:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Balanced Accuracy
  - PR-AUC

## 📂 Output Files
- `models/regression_model.joblib`
- `models/classification_model.joblib`
- `reports/model_metrics.json`
- `reports/segment_stats.json`

---

# 🌐 Milestone 4: Processing Time Estimator Engine

## ✅ Objectives
- Build prediction engine for real-time input

## 🔧 Tasks Completed
- Created `predict.py` pipeline:
  - Input preprocessing
  - Feature engineering
  - Encoding
- Integrated regression + classification models
- Generated:
  - Processing time prediction
  - Visa status prediction
  - Confidence score

---

# 💻 Milestone 5: Web App Development & Deployment

## ✅ Objectives
- Build user-friendly web interface
- Deploy application for real-world use

## 🔧 Features Implemented
- Streamlit-based UI (`app/main.py`)
- Input form:
  - Visa class
  - Country
  - Worksite
  - Application month
- Output:
  - Predicted processing time
  - Estimated range
  - Visa status + confidence
- Visualizations:
  - Monthly trends
  - Processing distribution

## 🔌 API Integration
- FastAPI endpoint (`/estimate`)
- JSON-based input/output support

---

# 📂 Project Structure
