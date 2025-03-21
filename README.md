
# Telco Customer Churn Prediction

A machine learning web application to predict customer churn in a telecommunications company using Flask and scikit-learn.

## Features

- **Single Customer Prediction**: Enter individual customer details to predict their churn probability
- **Batch Prediction**: Upload a CSV file to predict churn for multiple customers
- **Data Visualization**: 
  - Feature importance plots
  - Correlation analysis
  - Interactive visualizations
- **Model Performance Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

## Getting Started

1. Click the "Run" button to start the Flask application
2. Access the web interface through the provided URL
3. Choose between single prediction or batch prediction

## Usage

### Single Prediction
1. Navigate to "Single Prediction"
2. Fill in customer details
3. Submit to get churn probability

### Batch Prediction
1. Navigate to "Batch Prediction"
2. Upload a CSV file with customer data
3. Use the sample dataset provided if needed

## Data Requirements

The model expects the following features:
- Numeric: SeniorCitizen, MonthlyCharges, TotalCharges
- Categorical: gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

## Technology Stack

- Backend: Flask (Python)
- Frontend: Bootstrap, Chart.js
- ML: scikit-learn
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn

## Sample Data

A sample dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) is provided in the folder for testing the application.
