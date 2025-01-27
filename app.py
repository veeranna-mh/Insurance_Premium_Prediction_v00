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

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=66, step=1)
    height = st.number_input('Height (cm)', min_value=130, max_value=190, step=1)
    weight = st.number_input('Weight (kg)', min_value=41, max_value=160, step=1)
    diabetes = st.selectbox('Diabetes', [0, 1], help='0: No, 1: Yes')

with col2:
    blood_pressure = st.selectbox('Blood Pressure Problems', [0, 1], help='0: No, 1: Yes')
    transplants = st.selectbox('Any Transplants', [0, 1], help='0: No, 1: Yes')
    chronic_diseases = st.selectbox('Any Chronic Diseases', [0, 1], help='0: No, 1: Yes')
    allergies = st.selectbox('Known Allergies', [0, 1], help='0: No, 1: Yes')
    cancer_history = st.selectbox('History of Cancer in Family', [0, 1], help='0: No, 1: Yes')

# Input for the number of major surgeries
major_surgeries = st.number_input('Number of Major Surgeries', min_value=0, max_value=3, step=1)

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
