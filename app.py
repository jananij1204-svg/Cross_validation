import streamlit as st
import pandas as pd
import pickle

# =======================
# LOAD YOUR MODEL
# =======================
MODEL_PATH = "cross.pkl"

st.title("Prediction App using cross.pkl Model")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# =======================
# FEATURES USED BY MODEL
# IMPORTANT: Must match the model exactly
# =======================

# ❗❗ CHANGE THESE IF YOUR MODEL USED DIFFERENT FEATURES
FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

st.write("### Model expects these features:")
st.write(FEATURE_COLS)

# =======================
# USER INPUT FORM
# =======================
st.write("## Enter Passenger Details")

pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)

# Convert sex
sex_value = 0 if sex == "male" else 1

# Create DataFrame with EXACT model features
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_value,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare
}])

st.write("### Input to Model:")
st.dataframe(input_df)

# =======================
# PREDICTION
# =======================
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
