import streamlit as st
import pandas as pd
import pickle
import zipfile
import os

# Extract model ZIP once on app start
if not os.path.exists("model/random_forest_house_price_model.pkl"):
    with zipfile.ZipFile("random_forest_house_price_model.zip", 'r') as zip_ref:
        zip_ref.extractall("model")

# Load the model
with open("model/random_forest_house_price_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load dataset columns (may be a dict or list)
with open("dataset.pkl", 'rb') as f:
    data = pickle.load(f)

# If data is dict, get 'data_columns' key; else assume list
if isinstance(data, dict) and 'data_columns' in data:
    data_columns = data['data_columns']
else:
    data_columns = data

# Load locations
df = pd.read_csv("Bengaluru_House_Data.csv")
locations = sorted(df['location'].dropna().unique())

# Streamlit UI
st.title("🏠 Bengaluru House Price Predictor")
st.markdown("### Enter the details of the house:")

location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Total Square Feet", min_value=300, max_value=10000, step=10)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10)
bhk = st.number_input("Enter Number of Bedrooms (BHK)", min_value=1, max_value=10)

if st.button("Predict Price"):
    try:
        # Create input dict with zero for all columns
        input_dict = {col: 0 for col in data_columns}
        
        input_dict['total_sqft'] = sqft
        input_dict['bath'] = bath
        input_dict['bhk'] = bhk

        # One-hot encode location
        location_col = f"location_{location}"
        if location_col in input_dict:
            input_dict[location_col] = 1
        else:
            st.warning(f"⚠️ Location '{location}' not found in model columns. Prediction may be inaccurate.")

        # Convert to dataframe
        input_df = pd.DataFrame([input_dict])

        # Predict price
        prediction = model.predict(input_df)[0]

        # Format price to 2 decimals including trailing zeros
        formatted_price = f"{prediction:.2f}"

        st.success(f"Estimated Price: ₹{formatted_price} lakhs")

    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
