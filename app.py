import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ===========================
# Load Model
# ===========================
@st.cache_resource
def load_model():
    with open("/mnt/data/crossv.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Prediction App using crossv.pkl")
st.write("This app uses your uploaded model to make predictions.")

# ===========================
# Dynamic Input Based on Model
# ===========================

# Try to detect required number of input features
try:
    n_features = model.n_features_in_
except:
    st.error("Model does not provide 'n_features_in_'. Please tell me your feature names.")
    st.stop()

st.subheader("Enter Input Values")

# Create number inputs dynamically
inputs = []
for i in range(n_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

# Convert to numpy array
input_array = np.array(inputs).reshape(1, -1)

# ===========================
# Predict Button
# ===========================
if st.button("Predict"):
    try:
        prediction = model.predict(input_array)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
