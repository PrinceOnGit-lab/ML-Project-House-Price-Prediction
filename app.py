import streamlit as st
import pickle
import pandas as pd

# Load model
with open('random_forest_house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset (for dropdown options)
with open('dataset.pkl', 'rb') as f:
    df = pickle.load(f)

# Title
st.title('🏠 Bengaluru House Price Prediction App')

# Input fields with placeholders
location = st.selectbox('Select Location', ['-- Select Location --'] + sorted(df['location'].unique()))
total_sqft = st.text_input('Enter Total Square Feet (e.g., 1000)')
bath = st.selectbox('Number of Bathrooms', ['-- Select --', 1, 2, 3, 4, 5])
bhk = st.selectbox('Number of BHK', ['-- Select --', 1, 2, 3, 4, 5])

# Predict button
if st.button('Predict Price'):
    # Validation
    if (
        location == '-- Select Location --' or
        bath == '-- Select --' or
        bhk == '-- Select --' or
        total_sqft.strip() == ''
    ):
        st.warning('⚠️ Please select all fields and enter valid inputs.')
    else:
        try:
            total_sqft_val = float(total_sqft)
            input_df = pd.DataFrame([{
                'location': location,
                'total_sqft': total_sqft_val,
                'bath': int(bath),
                'BHK': int(bhk)
            }])

            prediction = model.predict(input_df)[0]
            st.success(f"🏷️ Estimated Price: ₹{round(prediction, 2)} lakhs")
        except ValueError:
            st.error("❌ Please enter a valid number for total square feet.")

