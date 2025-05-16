import streamlit as st
import pandas as pd
import pickle
import zipfile
import os

# --- Extract the model zip ---
with zipfile.ZipFile("random_forest_house_price_model.zip", 'r') as zip_ref:
    zip_ref.extractall("model")

# --- Load the model ---
model_path = "model/random_forest_house_price_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# --- Load column names for input vector ---
with open("dataset.pkl", 'rb') as f:
    data_columns = pickle.load(f)

# --- Load the original dataset for location list ---
df = pd.read_csv("Bengaluru_House_Data.csv")
locations = sorted(df['location'].dropna().unique())

# --- Streamlit UI ---
st.title("🏠 Bengaluru House Price Predictor")
st.markdown("Enter the house details below:")

location = st.selectbox("Location", locations)
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=10)
bath = st.slider("Number of Bathrooms", 1, 10, 2)
bhk = st.slider("Number of Bedrooms (BHK)", 1, 10, 3)

if st.button("Predict Price"):
    try:
        # Prepare input data
        input_data = pd.DataFrame(columns=data_columns)
        input_data.loc[0] = [0] * len(data_columns)

        input_data.at[0, 'total_sqft'] = sqft
        input_data.at[0, 'bath'] = bath
        input_data.at[0, 'bhk'] = bhk

        location_col = f"location_{location}"
        if location_col in data_columns:
            input_data.at[0, location_col] = 1

        # Predict
        prediction = model.predict(input_data)[0]
        st.success(f"🏷️ Estimated Price: ₹ {round(prediction, 2)} Lakhs")

    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
