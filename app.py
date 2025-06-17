import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data and model
@st.cache_data
def load_model_and_data():
    df = pd.read_csv("CO2 Emissions_Canada.csv")
    features = [
        'Engine Size(L)', 'Cylinders',
        'Fuel Consumption City (L/100 km)',
        'Fuel Consumption Hwy (L/100 km)',
        'Fuel Consumption Comb (L/100 km)'
    ]
    df_clean = df.dropna(subset=features + ['CO2 Emissions(g/km)'])
    df_clean['Car'] = df_clean['Make'].str.upper() + ' ' + df_clean['Model'].str.upper()
    car_specs = df_clean.groupby('Car')[features].mean().to_dict('index')

    X = df_clean[features]
    y = df_clean['CO2 Emissions(g/km)']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, car_specs, df_clean

# Load everything
model, car_specs, df_clean = load_model_and_data()

# Streamlit UI
st.title("CO₂ Emissions Predictor")
st.markdown("Select a car Make and Model to estimate CO₂ emissions.")

# Step 1: Get unique car makes
makes = sorted(df_clean['Make'].unique())

# Step 2: Select car make from dropdown
selected_make = st.selectbox("Select Car Make", makes)

# Step 3: Filter models based on selected make
models = sorted(df_clean[df_clean['Make'] == selected_make]['Model'].unique())
selected_model = st.selectbox("Select Car Model", models)

# Step 4: Predict button
if st.button("Predict"):
    car_key = f"{selected_make.upper()} {selected_model.upper()}"
    if car_key not in car_specs:
        st.error(f"Car model '{selected_make} {selected_model}' not found in the database.")
    else:
        features_input = car_specs[car_key]
        features_df = pd.DataFrame([features_input])
        prediction = model.predict(features_df)[0]
        st.subheader("Car Specifications")
        st.write(f"**Engine Size:** {features_input['Engine Size(L)']:.1f} L")
        st.write(f"**Cylinders:** {int(features_input['Cylinders'])}")
        st.write(f"**Fuel Consumption City:** {features_input['Fuel Consumption City (L/100 km)']:.1f} L/100 km")
        st.write(f"**Fuel Consumption Hwy:** {features_input['Fuel Consumption Hwy (L/100 km)']:.1f} L/100 km")
        st.write(f"**Fuel Consumption Comb:** {features_input['Fuel Consumption Comb (L/100 km)']:.1f} L/100 km")

        # Display prediction
        st.success(f"Predicted CO₂ Emission for {selected_make} {selected_model}: **{prediction:.2f} g/km**")

        # ✅ NEW: Show input as table
        st.subheader("🔍 Feature Summary Table")
        st.dataframe(features_df)

        # ✅ NEW: Metric box for emission
        st.metric(label="🌿 Estimated CO₂ Emission", value=f"{prediction:.2f} g/km")

        # ✅ NEW: Compare with average in dataset
        avg_emission = df_clean['CO2 Emissions(g/km)'].mean()
        delta = prediction - avg_emission
        comparison = "higher" if delta > 0 else "lower"
        st.info(f"This is {abs(delta):.2f} g/km {comparison} than the average CO₂ emission of {avg_emission:.2f} g/km.")

