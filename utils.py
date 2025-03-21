import pandas as pd
import numpy as np
import io
import base64
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(data):
    """
    Preprocess the input data to match the format expected by the model.
    The model expects one-hot encoded features in a specific order.
    """
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()

    # Drop any columns that are not part of the expected feature set
    if 'Unnamed: 0' in processed_data.columns:
        processed_data = processed_data.drop('Unnamed: 0', axis=1)

    # Check if 'customerID' is in the dataframe and drop it
    if 'customerID' in processed_data.columns:
        processed_data = processed_data.drop('customerID', axis=1)

    # Process the target variable separately (we will drop it before prediction)
    target = None
    if 'Churn' in processed_data.columns:
        # Convert Yes/No to 1/0 if needed
        if processed_data['Churn'].dtype == 'object':
            processed_data['Churn'] = processed_data['Churn'].map({
                'Yes': 1,
                'No': 0
            })
        # Save target for later use
        target = processed_data['Churn']

    # Define expected columns in the model's expected order
    expected_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
        'gender_Male', 'Partner_No', 'Partner_Yes', 'Dependents_No',
        'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
        'MultipleLines_No', 'MultipleLines_No phone service',
        'MultipleLines_Yes', 'InternetService_DSL',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service',
        'OnlineSecurity_Yes', 'OnlineBackup_No',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No internet service',
        'DeviceProtection_Yes', 'TechSupport_No',
        'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No internet service',
        'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
        'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
        'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72'
    ]

    # Check if data is already preprocessed (contains most one-hot encoded columns)
    encoded_cols = [col for col in processed_data.columns if '_' in col]

    # If the data is already one-hot encoded (like in tel_churn.csv)
    if len(encoded_cols
           ) >= 30:  # Arbitrary threshold for determining pre-encoded data
        print(
            "DEBUG: Data appears to be pre-encoded, using as-is with column alignment"
        )
        result_df = processed_data.copy()

        # Drop the target column if present (should not be an input feature)
        if 'Churn' in result_df.columns:
            result_df = result_df.drop('Churn', axis=1)

        # Convert any 'True'/'False' string values to 1/0 integers
        for col in result_df.columns:
            if result_df[col].dtype == 'object':
                # Check if the column contains boolean-like strings
                if set(result_df[col].dropna().unique()).issubset(
                    {'True', 'False', True, False}):
                    result_df[col] = result_df[col].map({
                        'True': 1,
                        'False': 0,
                        True: 1,
                        False: 0
                    })
                    print(f"DEBUG: Converted boolean strings in column {col}")

        # Make sure all numeric columns are correct type
        for col in ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                result_df[col] = result_df[col].fillna(0)

        # Check for any missing expected columns and add them with zeros
        for col in expected_columns:
            if col not in result_df.columns:
                result_df[col] = 0

        # Ensure columns are in the expected order
        # Use only columns that exist in expected_columns (handle extra columns in input)
        common_cols = [
            col for col in expected_columns if col in result_df.columns
        ]
        result_df = result_df[common_cols]

        print(
            f"DEBUG: First row of preprocessed data: {result_df.iloc[0].values}"
        )

        return result_df

    # Traditional preprocessing path for raw data:
    # Handle TotalCharges column (convert to numeric)
    if 'TotalCharges' in processed_data.columns and processed_data[
            'TotalCharges'].dtype == 'object':
        processed_data['TotalCharges'] = pd.to_numeric(
            processed_data['TotalCharges'], errors='coerce')
        processed_data['TotalCharges'] = processed_data['TotalCharges'].fillna(
            0)

    # Create tenure groups if tenure is present
    if 'tenure' in processed_data.columns:
        # Create tenure_group
        processed_data['tenure_group'] = pd.cut(
            processed_data['tenure'],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=[
                '1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'
            ])

        # Drop the original tenure column as it was not used in model training
        processed_data = processed_data.drop('tenure', axis=1)

    # Create result DataFrame with correct structure
    result_df = pd.DataFrame(index=processed_data.index)

    # Copy numeric columns
    for col in ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']:
        if col in processed_data.columns:
            result_df[col] = processed_data[col]
        else:
            result_df[col] = 0  # Default value if missing

    # Process categorical columns with explicit one-hot encoding
    categorical_mappings = {
        'gender': ['Female', 'Male'],
        'Partner': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
        'OnlineBackup': ['No', 'No internet service', 'Yes'],
        'DeviceProtection': ['No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No internet service', 'Yes'],
        'StreamingTV': ['No', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'No internet service', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['No', 'Yes'],
        'PaymentMethod': [
            'Bank transfer (automatic)', 'Credit card (automatic)',
            'Electronic check', 'Mailed check'
        ],
        'tenure_group':
        ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72']
    }

    # For each categorical feature
    for feature, categories in categorical_mappings.items():
        if feature in processed_data.columns:
            # For each possible category
            for category in categories:
                col_name = f"{feature}_{category}"
                # Set column to 1 if the category matches, 0 otherwise
                result_df[col_name] = (
                    processed_data[feature] == category).astype(int)
        else:
            # If feature is missing, set all category columns to 0
            for category in categories:
                col_name = f"{feature}_{category}"
                result_df[col_name] = 0

    # Ensure columns are in the expected order
    # Use only columns that exist in expected_columns (handle different versions of feature names)
    common_cols = [col for col in expected_columns if col in result_df.columns]
    result_df = result_df[common_cols]

    return result_df


def encode_categorical(data):
    """
    One-hot encode categorical features
    """
    # Define categorical features
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]

    # Only encode features that are in the dataframe
    categorical_features = [
        f for f in categorical_features if f in data.columns
    ]

    # One-hot encode categorical features
    encoded_data = pd.get_dummies(data,
                                  columns=categorical_features,
                                  drop_first=False)

    # Check if there are any missing columns compared to training data
    # The following prefixes should be present based on the model training data
    expected_prefixes = {
        'gender_': ['Female', 'Male'],
        'Partner_': ['No', 'Yes'],
        'Dependents_': ['No', 'Yes'],
        'PhoneService_': ['No', 'Yes'],
        'MultipleLines_': ['No', 'No phone service', 'Yes'],
        'InternetService_': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity_': ['No', 'No internet service', 'Yes'],
        'OnlineBackup_': ['No', 'No internet service', 'Yes'],
        'DeviceProtection_': ['No', 'No internet service', 'Yes'],
        'TechSupport_': ['No', 'No internet service', 'Yes'],
        'StreamingTV_': ['No', 'No internet service', 'Yes'],
        'StreamingMovies_': ['No', 'No internet service', 'Yes'],
        'Contract_': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling_': ['No', 'Yes'],
        'PaymentMethod_': [
            'Bank transfer (automatic)', 'Credit card (automatic)',
            'Electronic check', 'Mailed check'
        ],
        'tenure_group_':
        ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72']
    }

    # Add missing columns with zeros if they don't exist
    for prefix, values in expected_prefixes.items():
        for value in values:
            column = f"{prefix}{value}"
            if column not in encoded_data.columns:
                encoded_data[column] = 0

    return encoded_data


def create_feature_plot(x, y, title):
    """
    Create a horizontal bar plot of feature importances.
    Returns a base64-encoded PNG image string.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', dpi=300)
        indices = np.argsort(y)  # Sort for better readability
        x_sorted = [x[i] for i in indices]
        y_sorted = [y[i] for i in indices]
        
        ax.barh(x_sorted, y_sorted, color='skyblue')
        ax.set_xlabel('Importance Value', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        plt.tight_layout()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error in feature plot: {e}")
        plot_data = None
    finally:
        plt.close('all')
        buffer.close()
    return plot_data

def create_correlation_plot(data):
    """
    Create a correlation heatmap for numeric features.
    Returns a base64-encoded PNG image string or None if invalid.
    """
    try:
        # Select and clean numeric data
        numeric_data = data.select_dtypes(include=['number'])
        numeric_data = numeric_data.dropna(axis=1, how='all')
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Skip if insufficient numeric columns
        if numeric_data.shape[1] < 2:
            return None

        # Create correlation matrix
        corr_matrix = numeric_data.corr()

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white', dpi=300)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Feature Correlation', fontsize=14, pad=20)
        plt.tight_layout()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error in correlation plot: {e}")
        plot_data = None
    finally:
        plt.close('all')
        buffer.close()
    return plot_data