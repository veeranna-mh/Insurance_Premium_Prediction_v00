import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model and transformer
with open('random_forest_model.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    poly_degree2 = data['poly_transformer']

# Streamlit app logic
st.title('Insurance Premium Prediction')

# Input fields
age = st.number_input('Age', min_value=18, max_value=66, step=1)
height = st.number_input('Height (cm)', min_value=145, max_value=188, step=1)
weight = st.number_input('Weight (kg)', min_value=51, max_value=132, step=1)
diabetes = st.selectbox('Diabetes', [0, 1])
blood_pressure = st.selectbox('Blood Pressure Problems', [0, 1])
transplants = st.selectbox('Any Transplants', [0, 1])
chronic_diseases = st.selectbox('Any Chronic Diseases', [0, 1])
allergies = st.selectbox('Known Allergies', [0, 1])
cancer_history = st.selectbox('History of Cancer in Family', [0, 1])
major_surgeries = st.number_input('Number of Major Surgeries', min_value=0, max_value=3, step=1)

# Calculate BMI
bmi = weight / ((height / 100) ** 2)

# Make prediction
# if st.button('Predict Premium'):
#     input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases,
#                             height, weight, allergies, cancer_history, major_surgeries, bmi]])
#     input_poly = poly_degree2.transform(input_data)  # Transform input data
#     prediction = model.predict(input_poly)[0]
#     st.success(f'Estimated Insurance Premium: ${prediction:,.2f}')


if st.button('Predict Premium'):
    input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases,
                            height, weight, allergies, cancer_history, major_surgeries, bmi]])
    input_poly = poly_degree2.transform(input_data)  # Transform input data
    prediction = model.predict(input_poly)[0]

    # Convert prediction back to original scale
    min_val = 15000  # Replace with actual min value from your data
    max_val = 40000  # Replace with actual max value from your data
    original_prediction = (prediction * (max_val - min_val)) + min_val

    st.success(f'Estimated Insurance Premium: {original_prediction:,.2f}')
