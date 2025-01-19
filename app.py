import streamlit as st
import pickle
import numpy as np

# Load the saved Random Forest model
# model_filename = 'random_forest_model.pkl'
# with open(model_filename, 'rb') as file:
#     model = pickle.load(file)

# App title
st.title('Insurance Premium Prediction')

"""

# Input fields for user data
st.header('Enter the details to predict the insurance premium:')
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

# Make prediction when the user clicks the button
if st.button('Predict Premium'):
    # Prepare the input as a 2D array
    input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases,
                            height, weight, allergies, cancer_history, major_surgeries, bmi]])
    # Transform the input using PolynomialFeatures (if applicable)
    input_poly = poly_degree2.transform(input_data)

    # Make prediction
    prediction = model.predict(input_poly)[0]

    # Display the prediction
    st.success(f'Estimated Insurance Premium: ${prediction:,.2f}')

    
"""