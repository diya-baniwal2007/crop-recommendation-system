import streamlit as st
import joblib

# Load model and label encoder
model = joblib.load('crop_recommendation_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("🌾 Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    predicted_class = model.predict(input_data)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    st.success(f"🌱 Recommended Crop: {predicted_label}")

