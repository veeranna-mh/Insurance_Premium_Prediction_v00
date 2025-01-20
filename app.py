import streamlit as st
import pickle
import numpy as np

# Load the saved model and transformer
with open('random_forest_model.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    poly_degree2 = data['poly_transformer']

# App title
st.title('Insurance Premium Prediction')

# Two-column layout for inputs
col1, col2 = st.columns(2)

# Inputs in the first column
with col1:
    age = st.selectbox('Age', options=list(range(18, 67)))  # Dropdown for Age
    height = st.selectbox('Height (cm)', options=list(range(140, 190)))  # Dropdown for Height
    weight = st.selectbox('Weight (kg)', options=list(range(31, 150)))  # Dropdown for Weight
    diabetes = st.selectbox('Diabetes', options=[0, 1], help="0: No, 1: Yes")
    blood_pressure = st.selectbox('Blood Pressure Problems', options=[0, 1], help="0: No, 1: Yes")

# Inputs in the second column
with col2:
    transplants = st.selectbox('Any Transplants', options=[0, 1], help="0: No, 1: Yes")
    chronic_diseases = st.selectbox('Any Chronic Diseases', options=[0, 1], help="0: No, 1: Yes")
    allergies = st.selectbox('Known Allergies', options=[0, 1], help="0: No, 1: Yes")
    cancer_history = st.selectbox('History of Cancer in Family', options=[0, 1], help="0: No, 1: Yes")
    major_surgeries = st.selectbox('Number of Major Surgeries', options=list(range(0, 4)))  # Dropdown for Surgeries

# Calculate BMI
bmi = weight / ((height / 100) ** 2)

# Make prediction when the user clicks the button
if st.button('Predict Premium'):
    # Prepare the input as a 2D array
    input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases,
                            height, weight, allergies, cancer_history, major_surgeries, bmi]])
    # Transform the input using PolynomialFeatures
    input_poly = poly.transform(input_data)

    # Make prediction
    prediction = model.predict(input_poly)[0]

    # Convert prediction back to original scale
    min_val = 15000  # Replace with actual min value from your data
    max_val = 40000  # Replace with actual max value from your data
    original_prediction = (prediction * (max_val - min_val)) + min_val

    # Display the result
    st.success(f'Estimated Insurance Premium: {original_prediction:,.2f}')

