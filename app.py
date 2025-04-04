import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Function to load a file safely
def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's available.")
        return None

# Load model and dataset safely
model_path = 'random_forest_house_price_model.pkl'
dataset_path = 'dataset.pkl'

model = load_pickle_file(model_path)
dataset = load_pickle_file(dataset_path)

if model is None or dataset is None:
    st.stop()  # Stop execution if required files are missing

# Get unique locations
locations = dataset['location'].unique()

st.title("Bangalore House Price Prediction")

# UI inputs
location = st.selectbox('Location', locations)
total_sqft = st.number_input('Total Square Foot', min_value=0.0, value=1000.0)

col1, col2 = st.columns(2)
with col1:
    bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
with col2:
    bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, value=2)

# Predict button
if st.button('Predict'):
    input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'BHK'])
    try:
        prediction = model.predict(input_data)[0]
        st.write(f"The predicted price is {prediction:.2f} lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
