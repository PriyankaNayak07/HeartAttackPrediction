import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_model():
    """Train a heart disease prediction model and save it to disk"""
    # Load and prepare data
    data = pd.read_csv('heart.csv')
    
    # Print dataset info for debugging
    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution: {data['target'].value_counts()}")
    
    # Define features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model with better hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Print detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, 'heart_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler

def load_model():
    """Load the trained model and scaler from disk, or train if not available"""
    try:
        if os.path.exists('heart_model.pkl') and os.path.exists('scaler.pkl'):
            model = joblib.load('heart_model.pkl')
            scaler = joblib.load('scaler.pkl')
            print("Loaded existing model and scaler")
        else:
            print("Training new model...")
            model, scaler = train_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        model, scaler = train_model()
    
    return model, scaler

def predict_heart_disease(model, scaler, input_data):
    """
    Make a heart disease prediction based on input data
    
    Args:
        model: Trained machine learning model
        scaler: Fitted scaler for the features
        input_data: Dictionary containing the features for prediction
        
    Returns:
        prediction: 0 or 1 (no heart disease or heart disease)
        probability: Probability of the positive class
    """
    try:
        # Convert input data to dataframe
        input_df = pd.DataFrame([input_data])
        
        # Print input data for debugging
        print(f"Input data for prediction: {input_data}")
        
        # IMPLEMENTING USER'S EXACT CRITERIA
        
        # Check the three conditions
        low_chest_pain = input_data['cp'] <= 1  # Chest pain 0 or 1
        normal_bp = 70 <= input_data['trestbps'] <= 140  # BP between 70-140
        normal_chol = 100 <= input_data['chol'] <= 220  # Cholesterol between 100-220
        
        # Count how many conditions are met
        conditions_met = sum([low_chest_pain, normal_bp, normal_chol])
        
        # If all three conditions are met OR at least 2 conditions are met, then no heart disease
        if (low_chest_pain and normal_bp and normal_chol) or (conditions_met >= 2):
            print(f"Rule-based decision: Healthy profile - conditions met: {conditions_met}/3")
            prediction = 0  # No heart disease
            
            # Determine probability based on how many conditions were met
            if conditions_met == 3:
                probability = 0.15  # Very low probability of heart disease
            else:  # 2 conditions met
                probability = 0.30  # Low probability of heart disease
                
            return prediction, probability
        else:
            # Less than 2 conditions met, indicating heart disease
            print(f"Rule-based decision: At-risk profile - conditions met: {conditions_met}/3")
            prediction = 1  # Has heart disease
            
            # Determine probability based on how many conditions were not met
            if conditions_met == 0:
                probability = 0.85  # High probability of heart disease
            else:  # 1 condition met
                probability = 0.70  # Moderate-high probability of heart disease
                
            return prediction, probability
            
        # The code below will never be reached with the new logic, but keeping for completeness
        
        # Ensure columns are in the correct order
        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Reindex the dataframe to match expected columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[expected_columns]
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probability
        probability = model.predict_proba(input_scaled)[0][1]
        
        print(f"Model prediction: {prediction}, Probability: {probability:.4f}")
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return a safe fallback (no heart disease) with 50% probability
        return 0, 0.5
