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

# --- Load data columns (input feature structure) ---
with open("dataset.pkl", 'rb') as f:
    data_columns = pickle.load(f)

# --- Load original dataset for location list ---
df = pd.read_csv("Bengaluru_House_Data.csv")
locations = sorted(df['location'].dropna().unique())

# --- Streamlit UI ---
st.title("🏠 Bengaluru House Price Predictor")
st.markdown("### Enter the details of the house:")

location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Total Square Feet", min_value=300, max_value=10000, step=10)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10)
bhk = st.number_input("Enter Number of Bedrooms (BHK)", min_value=1, max_value=10)

if st.button("Predict Price"):
    try:
        # Prepare the input vector
        input_dict = {col: 0 for col in data_columns}
        input_dict['total_sqft'] = sqft
        input_dict['bath'] = bath
        input_dict['bhk'] = bhk

        # One-hot encode the location
        location_col = f"location_{location}"
        if location_col in input_dict:
            input_dict[location_col] = 1

        input_df = pd.DataFrame([input_dict])

        # Predict the price
        prediction = model.predict(input_df)[0]
            st.success(f"🏷️ Estimated Price: ₹{round(prediction, 2)} lakhs")

    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
