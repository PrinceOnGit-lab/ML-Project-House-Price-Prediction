import streamlit as st
import pandas as pd
import pickle
import os
import zipfile

# === Unzip the model file if not already done ===
zip_path = "random_forest_house_price_model.zip"
model_path = "random_forest_house_price_model.pkl"

if not os.path.exists(model_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        st.info("✅ Model extracted from zip.")
    else:
        st.error("❌ Zip file not found. Please upload 'random_forest_house_price_model.zip'.")
        st.stop()

# === Load model ===
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# === Load dataset for locations ===
try:
    with open("dataset.pkl", "rb") as f:
        df = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Dataset file not found. Please upload 'dataset.pkl'.")
    st.stop()

# === UI ===
st.title("🏠 Bengaluru House Price Prediction App")

location = st.selectbox("📍 Select Location", ["-- Select Location --"] + sorted(df["location"].unique()))
total_sqft = st.text_input("📐 Enter Total Square Feet (e.g., 1000)")
bath = st.selectbox("🛁 Number of Bathrooms", ["-- Select --", 1, 2, 3, 4, 5])
bhk = st.selectbox("🛏️ Number of BHK", ["-- Select --", 1, 2, 3, 4, 5])

# === Prediction ===
if st.button("🔍 Predict Price"):
    if (
        location == "-- Select Location --" or
        bath == "-- Select --" or
        bhk == "-- Select --" or
        total_sqft.strip() == ''
    ):
        st.warning("⚠️ Please complete all fields before predicting.")
    else:
        try:
            sqft_val = float(total_sqft)
            input_df = pd.DataFrame([{
                "location": location,
                "total_sqft": sqft_val,
                "bath": int(bath),
                "BHK": int(bhk)
            }])
            prediction = model.predict(input_df)[0]
            st.success(f"🏷️ Estimated Price: {round(prediction, 2)} lakh")
        except ValueError:
            st.error("❌ Please enter a valid number for square feet.")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
