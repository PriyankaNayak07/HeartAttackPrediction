import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from io import BytesIO
from model import load_model, predict_heart_disease
from utils import load_dataset, validate_inputs
from diet_recommendations import get_diet_recommendations

# Set page configuration
st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;  /* Light blue background */
        background-image: linear-gradient(to bottom right, #f0f8ff, #e6f7ff);
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #f0f8ff, #e6f7ff);
    }
    .heart-disease-positive {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid red;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0px;
    }
    .heart-disease-negative {
        background-color: rgba(0, 128, 0, 0.1);
        border-left: 5px solid green;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0px;
    }
    .result-title-positive {
        color: #cc0000;
        font-weight: bold;
    }
    .result-title-negative {
        color: #006600;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
model, scaler = load_model()

def generate_report(name, age, sex, chest_pain, blood_pressure, cholesterol, prediction, recommendations):
    """Generate a downloadable report with user data and recommendations"""
    report = f"""
    # Heart Health Assessment Report for {name}

    ## Personal Information
    - Name: {name}
    - Age: {age}
    - Sex: {sex}
    
    ## Clinical Data
    - Chest Pain Type: {chest_pain}
    - Resting Blood Pressure: {blood_pressure} mm Hg
    - Cholesterol Level: {cholesterol} mg/dl
    
    ## Assessment Result
    **{name} {'has indicators of heart disease' if prediction == 1 else 'does not have indicators of heart disease'}.**
    
    ## Recommendations
    {recommendations['general_guidelines']}
    
    ### Foods to Include:
    {chr(10).join(['- ' + item for item in recommendations['foods_to_include']])}
    
    ### Foods to Limit:
    {chr(10).join(['- ' + item for item in recommendations['foods_to_limit']])}
    
    ### Sample Meal Plan:
    {recommendations['sample_meal_plan']}
    
    *This report is for informational purposes only and is not a medical diagnosis. 
    Please consult with a healthcare professional for a comprehensive evaluation.*
    """
    
    return report

def get_download_link(report, filename="heart_health_report.txt", text="Download Report"):
    """Generate a download link for the report"""
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    # App title and description
    st.title("Heart Health Predictor")
    st.markdown("### Early Detection for Better Heart Health")
    
    # Force model retrain button (hidden in a sidebar expander for admin use)
    with st.sidebar.expander("Admin Options"):
        if st.button("Retrain Model"):
            import os
            if os.path.exists('heart_model.pkl'):
                os.remove('heart_model.pkl')
            if os.path.exists('scaler.pkl'):
                os.remove('scaler.pkl')
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Model will be retrained on next prediction")
            # Force refresh
            st.rerun()
    
    # Display healthy heart image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://pixabay.com/get/g734a0da1798ef84396bbcbb2340222614a583cec01a393838f71fa953263ce67d3686896adc8aa7a1fb2e664bf2300aedc448a04d8f660aebf16e2c8acca8d32_1280.jpg", 
                 width=300, caption="")

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Health Assessment", "Diet Recommendations"])
    
    with tab1:
        st.subheader("Personal Health Information")
        
        # Create form for user input
        with st.form(key="heart_disease_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name")
                age = st.number_input("Age", value=40)
                sex = st.selectbox("Sex", options=["Male", "Female"])
                
            
            with col2:
                chest_pain = st.selectbox("Chest Pain Type", 
                                         options=["0: No Pain", 
                                                  "1: Atypical Angina", 
                                                  "2: Non-anginal Pain", 
                                                  "3: Asymptomatic"])
                blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
                cholesterol = st.number_input("Cholesterol Level (mg/dl)", value=200)
            
            submit_button = st.form_submit_button(label="Predict Heart Health")
        
        # Process form submission
        if submit_button:
            # Validate inputs
            validation_result = validate_inputs(name, age, sex, blood_pressure, cholesterol)
            
            if validation_result["valid"]:
                # Process inputs for model
                sex_value = 1 if sex == "Male" else 0
                chest_pain_value = int(chest_pain.split(":")[0])
                
                # Create input data for prediction
                input_data = {
                    'age': age,
                    'sex': sex_value,
                    'cp': chest_pain_value,
                    'trestbps': blood_pressure,
                    'chol': cholesterol,
                    'fbs': 0,  # Default value
                    'restecg': 0,  # Default value
                    # Default values for other fields
                    'thalach': 150,  # Average max heart rate
                    'exang': 0,
                    'oldpeak': 0,
                    'slope': 1,
                    'ca': 0,
                    'thal': 2
                }
                
                # Display processing message
                with st.spinner("Analyzing your health information..."):
                    time.sleep(1)  # Simulate processing time
                    prediction, probability = predict_heart_disease(model, scaler, input_data)
                
                # Store prediction in session state for diet tab
                st.session_state["prediction"] = prediction
                st.session_state["name"] = name
                st.session_state["age"] = age
                st.session_state["sex"] = sex
                st.session_state["chest_pain"] = chest_pain
                st.session_state["blood_pressure"] = blood_pressure
                st.session_state["cholesterol"] = cholesterol
                
                # Get diet recommendations
                recommendations = get_diet_recommendations(
                    prediction,
                    age,
                    sex
                )
                
                # Store recommendations in session state
                st.session_state["recommendations"] = recommendations
                
                # Display prediction result
                st.subheader("Heart Health Assessment Results")
                
                # Create a custom container with light styling
                result_container = st.container()
                
                with result_container:
                    if prediction == 1:
                        st.markdown("""
                        <div class="heart-disease-positive">
                            <h4 class="result-title-positive">❗ Heart Disease Detected</h4>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<p><strong>{name}</strong>, our analysis indicates you have heart disease.</p>", unsafe_allow_html=True)
                        st.markdown("<h3>Recommendation:</h3>", unsafe_allow_html=True)
                        st.markdown("""
                        <ul>
                            <li>Please consult with a healthcare professional for a comprehensive evaluation</li>
                            <li>Check the Diet Recommendations tab for dietary guidelines</li>
                            <li>Regular monitoring of blood pressure and cholesterol is advised</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="heart-disease-negative">
                            <h4 class="result-title-negative">✅ No Heart Disease Detected</h4>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<p><strong>{name}</strong>, our analysis indicates you do not have heart disease.</p>", unsafe_allow_html=True)
                        st.markdown("<h3>Recommendation:</h3>", unsafe_allow_html=True)
                        st.markdown("""
                        <ul>
                            <li>Continue maintaining a healthy lifestyle</li>
                            <li>Check the Diet Recommendations tab for dietary guidelines to maintain heart health</li>
                            <li>Regular check-ups are still recommended for preventive care</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Generate report
                report = generate_report(
                    name, 
                    age, 
                    sex, 
                    chest_pain, 
                    blood_pressure, 
                    cholesterol, 
                    prediction, 
                    recommendations
                )
                
                # Display download link
                st.markdown("### Download Your Report")
                st.markdown(get_download_link(report), unsafe_allow_html=True)
            else:
                # Display validation errors
                st.error(validation_result["message"])
    
    with tab2:
        # Check if prediction has been made
        if "prediction" in st.session_state:
            st.subheader(f"Diet Recommendations for {st.session_state.get('name', 'You')}")
            
            # Display healthy diet image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("https://pixabay.com/get/g33273944d6f5d0f6eff09054ea4b6490b7eb36b86f2b789d2cef6d0353fd8809ba53e4023aff25cf4b9596c2584aa8277adcb8cc405d9e158723db5b1225d61e_1280.jpg", 
                         width=400, caption="Heart-Healthy Foods")
            
            # Get diet recommendations based on prediction
            if "recommendations" in st.session_state:
                recommendations = st.session_state["recommendations"]
            else:
                recommendations = get_diet_recommendations(
                    st.session_state["prediction"],
                    st.session_state["age"],
                    st.session_state["sex"]
                )
            
            # Display recommendations
            st.markdown("### Dietary Guidelines")
            st.markdown(recommendations["general_guidelines"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Foods to Include")
                for item in recommendations["foods_to_include"]:
                    st.markdown(f"- {item}")
            
            with col2:
                st.markdown("### Foods to Limit")
                for item in recommendations["foods_to_limit"]:
                    st.markdown(f"- {item}")
            
            st.markdown("### Sample Meal Plan")
            st.markdown(recommendations["sample_meal_plan"])
            
            st.info("These recommendations are general guidelines. Please consult with a nutritionist or healthcare provider for personalized advice.")
            
            # Display download report button
            if all(key in st.session_state for key in ["name", "age", "sex", "chest_pain", "blood_pressure", "cholesterol"]):
                report = generate_report(
                    st.session_state["name"], 
                    st.session_state["age"], 
                    st.session_state["sex"], 
                    st.session_state["chest_pain"], 
                    st.session_state["blood_pressure"], 
                    st.session_state["cholesterol"], 
                    st.session_state["prediction"], 
                    recommendations
                )
                
                st.markdown("### Download Your Report")
                st.markdown(get_download_link(report), unsafe_allow_html=True)
        else:
            st.info("Please complete the Health Assessment to receive personalized diet recommendations.")
            
            # Display a general healthy diet image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("https://pixabay.com/get/ge254de007263a17f7642038b3843881b22462a77a554c3df07dbcd579c2e53ef4624f1fe2eb71a9f3df4bc72a31a62d7162a325980a32391eabea864569214a3_1280.jpg", 
                         width=400, caption="Heart-Healthy Diet")
            
            st.markdown("### General Heart Health Diet Tips")
            st.markdown("""
            - Eat plenty of fruits, vegetables, and whole grains
            - Choose lean proteins like fish, poultry, and legumes
            - Limit saturated fats, trans fats, and cholesterol
            - Reduce sodium intake to help control blood pressure
            - Moderate alcohol consumption
            - Stay physically active and maintain a healthy weight
            """)

if __name__ == "__main__":
    main()
