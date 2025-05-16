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

# --- Load data columns (used for feature vector structure) ---
with open("dataset.pkl", 'rb') as f:
    data_columns = pickle.load(f)  # typically a list of column names

# --- Load original dataset to extract location list ---
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
        # Create input vector initialized to 0s
        input_dict = {col: 0 for col in data_columns}
        input_dict['total_sqft'] = sqft
        input_dict['bath'] = bath
        input_dict['bhk'] = bhk

        # Set location one-hot
        location_col = f"location_{location}"
        if location_col in input_dict:
            input_dict[location_col] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"🏷️ Estimated Price: ₹ {round(prediction, 2)} Lakhs")

    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
