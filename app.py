import streamlit as st
import pandas as pd
import pickle
import zipfile
import os

# Unzip the model file
with zipfile.ZipFile("random_forest_house_price_model.zip", 'r') as zip_ref:
    zip_ref.extractall("model")

# Load model and dataset
with open("model/random_forest_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("dataset.pkl", 'rb') as f:
    data_columns = pickle.load(f)  # Usually contains column names, like 'data_columns' key

# Load CSV to get location list
df = pd.read_csv("Bengaluru_House_Data.csv")

# Extract location names from dataset
locations = sorted(df['location'].dropna().unique())

# Streamlit UI
st.title("🏠 Bengaluru House Price Predictor")

st.markdown("### Enter the house details below:")

location = st.selectbox("Location", locations)
sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, step=50)
bath = st.slider("Bathrooms", 1, 10, 2)
bhk = st.slider("BHK (Bedrooms)", 1, 10, 3)

if st.button("Predict Price"):
    try:
        # Create input vector with the same order as training data
        input_data = pd.DataFrame(columns=data_columns)
        input_data.loc[0] = [0] * len(data_columns)

        input_data.at[0, 'total_sqft'] = sqft
        input_data.at[0, 'bath'] = bath
        input_data.at[0, 'bhk'] = bhk

        location_col = f"location_{location}"
        if location_col in data_columns:
            input_data.at[0, location_col] = 1

        prediction = model.predict(input_data)[0]
        st.success(f"🏷️ Estimated Price: ₹ {round(prediction, 2)} Lakhs")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
