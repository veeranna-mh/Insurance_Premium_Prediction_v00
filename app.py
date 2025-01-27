import streamlit as st
import pickle
import numpy as np

# Load the saved model and transformer
with open('rf_best_poly_model_for_streamlit.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    poly = data['poly_transformer']

# Streamlit app title
st.title('Insurance Premium Prediction App')

# Two-column layout for dropdown inputs
col1, col2 = st.columns(2)

with col1:
    # Dropdown for Age
    age = st.selectbox('Age', options=list(range(18, 67)), help="Select the age (18-66).")
    # Dropdown for Height
    height = st.selectbox('Height (cm)', options=list(range(130, 191)), help="Select height in cm (130-190).")
    # Dropdown for Weight
    weight = st.selectbox('Weight (kg)', options=list(range(41, 161)), help="Select weight in kg (41-160).")
    # Dropdown for Diabetes
    diabetes = st.selectbox('Diabetes', options=[0, 1], help="0: No, 1: Yes.")

with col2:
    # Dropdown for Blood Pressure Problems
    blood_pressure = st.selectbox('Blood Pressure Problems', options=[0, 1], help="0: No, 1: Yes.")
    # Dropdown for Any Transplants
    transplants = st.selectbox('Any Transplants', options=[0, 1], help="0: No, 1: Yes.")
    # Dropdown for Chronic Diseases
    chronic_diseases = st.selectbox('Any Chronic Diseases', options=[0, 1], help="0: No, 1: Yes.")
    # Dropdown for Known Allergies
    allergies = st.selectbox('Known Allergies', options=[0, 1], help="0: No, 1: Yes.")
    # Dropdown for History of Cancer in Family
    cancer_history = st.selectbox('History of Cancer in Family', options=[0, 1], help="0: No, 1: Yes.")

# Dropdown for Number of Major Surgeries
major_surgeries = st.selectbox('Number of Major Surgeries', options=list(range(0, 4)), help="Select the number of major surgeries (0-3).")

# When user clicks the "Predict" button
if st.button('Predict Premium'):
    # Prepare the input data
    input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases,
                            height, weight, allergies, cancer_history, major_surgeries]])
    
    # Transform input data using the PolynomialFeatures transformer
    input_poly = poly.transform(input_data)

    # Make prediction using the Random Forest model
    prediction = model.predict(input_poly)[0]

    # Display the prediction
    st.success(f'Estimated Insurance Premium: ${prediction:,.2f}')
