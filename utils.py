import pandas as pd
import re

def load_dataset():
    """Load and return the heart disease dataset"""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def validate_inputs(name, age, sex, blood_pressure, cholesterol):
    """
    Validate user inputs for the heart disease prediction form
    
    Args:
        name: User's name
        age: User's age
        sex: User's sex (Male/Female)
        blood_pressure: Resting blood pressure value
        cholesterol: Cholesterol level
        
    Returns:
        Dictionary with validation result and message
    """
    # Check if name is provided
    if not name:
        return {"valid": False, "message": "Please enter your name."}
    
    # Accept any name (removed restrictions on characters)
    
    # Accept any age (removed restrictions on age range)
    
    # Accept any blood pressure (removed restrictions on range)
    
    # Accept any cholesterol level (removed restrictions on range)
    
    # All validations passed
    return {"valid": True, "message": "All inputs are valid."}

def format_prediction_result(prediction, probability, name):
    """
    Format the prediction result into a human-readable message
    
    Args:
        prediction: Model prediction (0 or 1)
        probability: Prediction probability
        name: User's name
        
    Returns:
        Formatted message string
    """
    if prediction == 1:
        return {
            "status": "warning",
            "message": f"{name}, our analysis indicates you may have a {probability:.1%} risk of heart disease.",
            "details": "Please consult with a healthcare professional for a comprehensive evaluation."
        }
    else:
        return {
            "status": "success",
            "message": f"{name}, our analysis indicates a {(1-probability):.1%} probability that you do not have heart disease.",
            "details": "Continue maintaining a healthy lifestyle and attend regular check-ups."
        }
