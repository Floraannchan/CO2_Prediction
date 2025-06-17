import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("CO2 Emissions_Canada.csv")
    # Clean column names exactly as done during training
    df.columns = (
        df.columns.str.replace(" \(L/100 km\)", "", regex=True)
        .str.replace("\(L\)", "", regex=True)
        .str.replace("\(g/km\)", "", regex=True)
        .str.replace(" ", "_")
        .str.lower()
    )
    return df

# Load model and data
model = load_model()
df = load_data()

# Infer exact feature list used during training
X_train = df.drop(columns=["co2_emissions"], errors='ignore')
X_numeric = X_train.select_dtypes(include=['number'])
trained_features = X_numeric.columns.tolist()

# UI
st.title("CO₂ Emission Predictor")
st.markdown("Input vehicle specifications to get CO₂ emissions using a Random Forest model.")

# Dynamic input form for all trained features
user_input = {}
with st.form("vehicle_form"):
    st.subheader("Vehicle Specifications:")
    for feature in trained_features:
        default_value = float(df[feature].mean())
        label = feature.replace("_", " ").replace("(mpg)", "(MPG)").title()
        user_input[feature] = st.number_input(f"{label}:", value=default_value)
    submitted = st.form_submit_button("Predict CO₂ Emission")

# Predict and display result
if submitted:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated CO₂ Emission: **{prediction:.2f} g/km**")


