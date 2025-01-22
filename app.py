import streamlit as st
import pickle
import numpy as np

# Load the saved model and transformer
with open('random_forest_with_poly1.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    poly = data['poly_transformer']

# Streamlit App
st.title('Insurance Premium Prediction')

# Two-column layout for inputs
col1, col2 = st.columns(2)

# Inputs in the first column
with col1:
    age = st.selectbox('Age', options=list(range(18, 67)), help="Select the age (18-66).")
    height = st.selectbox('Height (cm)', options=list(range(140, 190)), help="Select height in cm (145-188).")
    weight = st.selectbox('Weight (kg)', options=list(range(41, 133)), help="Select weight in kg (51-132).")
    diabetes = st.selectbox('Diabetes', options=[0, 1], help="0: No, 1: Yes.")
    blood_pressure = st.selectbox('Blood Pressure Problems', options=[0, 1], help="0: No, 1: Yes.")

# Inputs in the second column
with col2:
    transplants = st.selectbox('Any Transplants', options=[0, 1], help="0: No, 1: Yes.")
    chronic_diseases = st.selectbox('Any Chronic Diseases', options=[0, 1], help="0: No, 1: Yes.")
    allergies = st.selectbox('Known Allergies', options=[0, 1], help="0: No, 1: Yes.")
    cancer_history = st.selectbox('History of Cancer in Family', options=[0, 1], help="0: No, 1: Yes.")
    major_surgeries = st.selectbox('Number of Major Surgeries', options=list(range(0, 4)), help="Select the number of major surgeries (0-3).")

# # Calculate BMI
# bmi = weight / ((height / 100) ** 2)

# Make prediction when the user clicks the button
if st.button('Predict Premium'):
    # Prepare the input data
    input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases,
                            height, weight, allergies, cancer_history, major_surgeries]])
    
    # Transform the input data using the loaded PolynomialFeatures transformer
    input_poly = poly.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_poly)[0]
    
    # Display the result
    st.success(f'Estimated Insurance Premium: {prediction:,.2f}')

