import streamlit as st
import pandas as pd
import numpy as np
import joblib  # for loading your pre-trained model

# Load your pre-trained model
model = joblib.load('best_model.pkl')

# Title of the web app
st.title("Autism Prediction Web App")

# Input section
st.header("Enter the details for Autism Prediction")

# Numeric questions (A1–A10)
A1 = st.number_input("A1 Score", min_value=0, max_value=10, step=1)
A2 = st.number_input("A2 Score", min_value=0, max_value=10, step=1)
A3 = st.number_input("A3 Score", min_value=0, max_value=10, step=1)
A4 = st.number_input("A4 Score", min_value=0, max_value=10, step=1)
A5 = st.number_input("A5 Score", min_value=0, max_value=10, step=1)
A6 = st.number_input("A6 Score", min_value=0, max_value=10, step=1)
A7 = st.number_input("A7 Score", min_value=0, max_value=10, step=1)
A8 = st.number_input("A8 Score", min_value=0, max_value=10, step=1)
A9 = st.number_input("A9 Score", min_value=0, max_value=10, step=1)
A10 = st.number_input("A10 Score", min_value=0, max_value=10, step=1)

# Demographic and medical inputs
age = st.number_input("Age", min_value=1, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "White", "Hispanic", "Other"])
jaundice = st.selectbox("Jaundice History", ["Yes", "No"])
country_of_res = st.selectbox("Country of Residence", ["India", "USA", "UK", "Other"])
used_app_before = st.selectbox("Used Apps Before", ["Yes", "No"])

# Newly added features
autism = st.selectbox("Family Member with Autism", ["Yes", "No"])
relation = st.selectbox("Relation of Respondent", ["Self", "Parent", "Relative", "Health Care Professional", "Others"])
result = st.number_input("Screening Score (e.g., AQ Score)", min_value=0, max_value=20, step=1)

# Convert categorical variables to numerical (consistent with training)
gender = 1 if gender == "Male" else 0
ethnicity = {"Asian": 1, "Black": 2, "White": 3, "Hispanic": 4, "Other": 5}[ethnicity]
jaundice = 1 if jaundice == "Yes" else 0
country_of_res = {"India": 1, "USA": 2, "UK": 3, "Other": 4}[country_of_res]
used_app_before = 1 if used_app_before == "Yes" else 0
autism = 1 if autism == "Yes" else 0
relation = {"Self": 1, "Parent": 2, "Relative": 3, "Health Care Professional": 4, "Others": 5}[relation]

# Combine all input features (now 19 total)
input_features = np.array([
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
    age, gender, ethnicity, jaundice, country_of_res,
    used_app_before, autism, relation, result
]).reshape(1, -1)

# Prediction button
if st.button("Predict Autism"):
    prediction = model.predict(input_features)
    result_label = prediction[0]

    # Display result
    if result_label == 1:
        st.subheader("Prediction: The person is likely to have Autism (ASD).")
    else:
        st.subheader("Prediction: The person is likely not to have Autism (ASD).")

    # Show probabilities if supported
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(input_features)
        st.write(f"Probability of ASD: {prob[0][1]:.2f}")
        st.write(f"Probability of Non-ASD: {prob[0][0]:.2f}")
