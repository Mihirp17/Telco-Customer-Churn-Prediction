import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
import logging
from utils import preprocess_data, create_feature_plot, create_correlation_plot
import io
import csv
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "telco-churn-prediction-secret")

# Load the trained model
try:
    model_path = "finalized_model.sav"
    logger.info(f"Attempting to load model from {model_path}")
    model = pickle.load(open(model_path, "rb"))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/single_prediction', methods=['GET', 'POST'])
def single_prediction():
    """Handle single prediction form."""
    if request.method == 'POST':
        try:
            # Extract form data
            data = {}
            
            # Extract and cast numeric fields
            data['SeniorCitizen'] = int(request.form.get('SeniorCitizen', 0))
            data['MonthlyCharges'] = float(request.form.get('MonthlyCharges', 0))
            data['TotalCharges'] = float(request.form.get('TotalCharges', 0))
            
            # Extract categorical fields
            data['gender'] = request.form.get('gender')
            data['Partner'] = request.form.get('Partner')
            data['Dependents'] = request.form.get('Dependents')
            data['PhoneService'] = request.form.get('PhoneService')
            data['MultipleLines'] = request.form.get('MultipleLines', 'No phone service') if request.form.get('PhoneService') == 'No' else request.form.get('MultipleLines')
            data['InternetService'] = request.form.get('InternetService')
            data['OnlineSecurity'] = request.form.get('OnlineSecurity', 'No internet service') if request.form.get('InternetService') == 'No' else request.form.get('OnlineSecurity')
            data['OnlineBackup'] = request.form.get('OnlineBackup', 'No internet service') if request.form.get('InternetService') == 'No' else request.form.get('OnlineBackup')
            data['DeviceProtection'] = request.form.get('DeviceProtection', 'No internet service') if request.form.get('InternetService') == 'No' else request.form.get('DeviceProtection')
            data['TechSupport'] = request.form.get('TechSupport', 'No internet service') if request.form.get('InternetService') == 'No' else request.form.get('TechSupport')
            data['StreamingTV'] = request.form.get('StreamingTV', 'No internet service') if request.form.get('InternetService') == 'No' else request.form.get('StreamingTV')
            data['StreamingMovies'] = request.form.get('StreamingMovies', 'No internet service') if request.form.get('InternetService') == 'No' else request.form.get('StreamingMovies')
            data['Contract'] = request.form.get('Contract')
            data['PaperlessBilling'] = request.form.get('PaperlessBilling')
            data['PaymentMethod'] = request.form.get('PaymentMethod')
            data['tenure'] = int(request.form.get('tenure', 0))
            
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Preprocess data
            processed_data = preprocess_data(df)
            
            if model is not None:
                # Make prediction
                predictions_proba = model.predict_proba(processed_data)[0]  # Get all probabilities
                prediction_proba = predictions_proba[1]  # Second value is probability of churn (class 1)
                prediction = 1 if prediction_proba > 0.5 else 0
                
                # Print debug information
                print(f"DEBUG - Single Prediction: All Probabilities: {predictions_proba}")
                print(f"DEBUG - Single Prediction: Churn Probability: {prediction_proba * 100}%")
                print(f"DEBUG - Single Prediction: Prediction: {prediction}")
                
                # Get feature importances
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    features = processed_data.columns[indices]
                    
                    # Get top 10 features
                    top_features = features[:10]
                    top_importances = importances[indices][:10]
                    
                    # Create feature importance plot
                    feature_plot = create_feature_plot(top_features, top_importances, 'Feature Importance')
                else:
                    feature_plot = None
                
                # Create correlation plot for numerical features
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_columns) > 1:  # Need at least 2 numeric columns for correlation
                    correlation_plot = create_correlation_plot(df[numeric_columns])
                else:
                    correlation_plot = None
                
                return render_template('prediction_results.html', 
                                      prediction=prediction,
                                      probability=prediction_proba * 100,
                                      feature_plot=feature_plot,
                                      correlation_plot=correlation_plot,
                                      single_prediction=True)
            else:
                flash("Model is not loaded. Cannot make prediction.", "danger")
                return redirect(url_for('single_prediction'))
                
        except Exception as e:
            logger.error(f"Error in single prediction: {e}")
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('single_prediction'))
    
    return render_template('single_prediction.html')

@app.route('/batch_prediction', methods=['GET', 'POST'])
def batch_prediction():
    """Handle batch prediction from uploaded CSV file."""
    if request.method == 'POST':
        try:
            # Check if a file was uploaded
            if 'file' not in request.files:
                flash("No file part", "danger")
                return redirect(request.url)
            
            file = request.files['file']
            
            # Check if the file is empty
            if file.filename == '':
                flash("No selected file", "danger")
                return redirect(request.url)
            
            # Check if the file is CSV
            if not file.filename.endswith('.csv'):
                flash("Only CSV files are allowed", "danger")
                return redirect(request.url)
            
            # Read file content
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_input = csv.reader(stream)
            
            # Convert to DataFrame
            header = next(csv_input)
            df = pd.DataFrame([row for row in csv_input], columns=header)
            
            # Process the data
            try:
                # Convert numeric columns to appropriate types
                if 'SeniorCitizen' in df.columns:
                    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
                
                if 'MonthlyCharges' in df.columns:
                    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
                
                if 'TotalCharges' in df.columns:
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                
                if 'tenure' in df.columns:
                    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
                
                # Preprocess data
                processed_data = preprocess_data(df)
                
                if model is not None:
                    # Make predictions
                    predictions_proba = model.predict_proba(processed_data)[:, 1]  # Second column is probability of churn (class 1)
                    predictions = [1 if prob > 0.5 else 0 for prob in predictions_proba]
                    
                    # Add predictions to original data
                    df['Churn_Prediction'] = predictions
                    df['Churn_Probability'] = predictions_proba * 100
                    
                    # Print debug information
                    print(f"DEBUG - Batch Prediction: First 5 probabilities: {predictions_proba[:5]}")
                    print(f"DEBUG - Batch Prediction: First 5 predictions: {predictions[:5]}")
                    print(f"DEBUG - Batch Prediction: Churn Rate: {(sum(predictions) / len(predictions)) * 100}%")
                    
                    # Calculate model performance metrics if actual churn data is available
                    model_metrics = {}
                    if 'Churn' in df.columns:
                        try:
                            # Make a copy to avoid modifying original data
                            churn_data = df['Churn'].copy()
                            
                            # Convert Churn column to numeric if it's not already
                            if churn_data.dtype == 'object':
                                churn_data = churn_data.map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0})
                            else:
                                churn_data = churn_data.astype(float)
                                
                            # Handle missing values - drop rows with NaN in Churn column
                            valid_indices = ~churn_data.isna()
                            if valid_indices.sum() < len(churn_data):
                                print(f"DEBUG - Found {len(churn_data) - valid_indices.sum()} rows with missing Churn values. These will be excluded from evaluation.")
                                
                            # Filter out NaN values for evaluation
                            actual = churn_data[valid_indices].values.astype(int)
                            predicted_filtered = [predictions[i] for i in range(len(predictions)) if valid_indices.iloc[i]]
                            
                            # Check if we have enough data for evaluation
                            if len(actual) == 0 or len(predicted_filtered) == 0:
                                print("DEBUG - Not enough valid data for evaluation after filtering NaN values")
                                raise ValueError("Not enough valid data for evaluation")
                                
                            # Calculate metrics
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                            
                            model_metrics['accuracy'] = accuracy_score(actual, predicted_filtered) * 100
                            model_metrics['precision'] = precision_score(actual, predicted_filtered, zero_division=0) * 100
                            model_metrics['recall'] = recall_score(actual, predicted_filtered, zero_division=0) * 100
                            model_metrics['f1'] = f1_score(actual, predicted_filtered, zero_division=0) * 100
                            
                            # Create confusion matrix
                            cm = confusion_matrix(actual, predicted_filtered)
                            true_neg, false_pos, false_neg, true_pos = cm.ravel()
                            
                            model_metrics['true_negatives'] = true_neg
                            model_metrics['false_positives'] = false_pos
                            model_metrics['false_negatives'] = false_neg
                            model_metrics['true_positives'] = true_pos
                            
                            print(f"DEBUG - Model Evaluation: Accuracy: {model_metrics['accuracy']}%")
                            print(f"DEBUG - Model Evaluation: Precision: {model_metrics['precision']}%")
                            print(f"DEBUG - Model Evaluation: Recall: {model_metrics['recall']}%")
                            print(f"DEBUG - Model Evaluation: F1 Score: {model_metrics['f1']}%")
                            print(f"DEBUG - Model Evaluation: Confusion Matrix: {cm}")
                        except Exception as e:
                            print(f"DEBUG - Error evaluating model: {e}")
                            # Continue without model metrics
                            pass
                    
                    # Convert DataFrame to HTML table
                    table_html = df.to_html(classes='table table-striped table-hover')
                    
                    # Calculate churn rate
                    churn_rate = (sum(predictions) / len(predictions)) * 100
                    
                    # Get feature importances
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        features = processed_data.columns[indices]
                        
                        # Get top 10 features
                        top_features = features[:10]
                        top_importances = importances[indices][:10]
                        
                        # Create feature importance plot
                        feature_plot = create_feature_plot(top_features, top_importances, 'Feature Importance')
                    else:
                        feature_plot = None
                    
                    # Create correlation plot for numerical features
                    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                    correlation_plot = create_correlation_plot(df[numeric_columns]) if len(numeric_columns) > 0 else None
                    
                    return render_template('prediction_results.html', 
                                          table=table_html,
                                          churn_rate=churn_rate,
                                          feature_plot=feature_plot,
                                          correlation_plot=correlation_plot,
                                          batch_prediction=True,
                                          model_metrics=model_metrics if 'accuracy' in model_metrics else None)
                else:
                    flash("Model is not loaded. Cannot make prediction.", "danger")
                    return redirect(url_for('batch_prediction'))
                    
            except Exception as e:
                logger.error(f"Error processing data: {e}")
                flash(f"Error processing data: {str(e)}", "danger")
                return redirect(url_for('batch_prediction'))
                
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('batch_prediction'))
    
    return render_template('batch_prediction.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
