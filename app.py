import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Load encoders if used
encoders = joblib.load("label_encoders.pkl")

# Streamlit App UI
st.title("Car Sales Price Prediction")

# User Input Form
annual_inc = st.number_input("Annual Income", min_value=5000, max_value=5000000, step=1000)
company = st.selectbox("Company", encoders['Company'].classes_)
model_name = st.selectbox("Model", encoders['Model'].classes_)
transmission = st.selectbox("Transmission", encoders['Transmission'].classes_)
color = st.selectbox("Color", encoders['Color'].classes_)

# Predict Button
if st.button("Predict Price"):
    # Encode inputs
    company_encoded = encoders['Company'].transform([company])[0]
    model_encoded = encoders['Model'].transform([model_name])[0]
    transmission_encoded = encoders['Transmission'].transform([transmission])[0]
    color_encoded = encoders['Color'].transform([color])[0]

    # Prepare input data
    input_data = pd.DataFrame([[annual_inc, company_encoded, model_encoded, transmission_encoded, color_encoded]],
                              columns=['Annual Inc', 'Company', 'Model', 'Transmission', 'Color'])

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Display Prediction
    st.success(f"Predicted Car Price: ${predicted_price:,.2f}")
