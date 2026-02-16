import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Trail Event Capacity & Risk Simulator", layout="wide")

st.title(" Trail Event Capacity & Risk Simulator")

# =========================
# CARGAR MODELOS
# =========================

@st.cache_resource
def load_models():
    rf_model = joblib.load("dnf_rf_model.joblib")
    log_model = joblib.load("high_dnf_log_model.joblib")
    return rf_model, log_model

rf_model, log_model = load_models()

# =========================
# INPUTS
# =========================

st.header("Event Parameters")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", min_value=1.0, value=50.0)
    elevation_gain = st.number_input("Elevation Gain (m)", min_value=0.0, value=2500.0)

with col2:
    participants = st.number_input("Number of Participants", min_value=1, value=500)
    year = st.number_input("Year", min_value=2014, max_value=2030, value=2025)

registration_fee = st.number_input("Registration Fee ($)", min_value=0.0, value=100.0)
fixed_cost = st.number_input("Estimated Fixed Cost ($)", min_value=0.0, value=20000.0)

# =========================
# FEATURE ENGINEERING
# =========================

elevation_per_km = elevation_gain / distance

features = pd.DataFrame([{
    "Distance": distance,
    "Elevation Gain": elevation_gain,
    "elevation_per_km": elevation_per_km,
    "N Participants": participants,
    "Year": year
}])

# =========================
# PREDICCIONES
# =========================

if st.button("Simulate Event"):

    predicted_dnf = rf_model.predict(features)[0]
    high_risk_prob = log_model.predict_proba(features)[0][1]

    revenue = participants * registration_fee
    profit = revenue - fixed_cost

    st.subheader(" Event Results")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Predicted DNF Rate", f"{predicted_dnf:.2%}")

    with colB:
        st.metric("High Risk Probability", f"{high_risk_prob:.2%}")

    with colC:
        st.metric("Estimated Profit ($)", f"${profit:,.0f}")

    st.divider()

    # Simple Risk Indicator
    if high_risk_prob > 0.7:
        st.error("⚠ High Risk Event – Consider capacity control and safety measures.")
    elif high_risk_prob > 0.4:
        st.warning("⚠ Moderate Risk – Monitor logistics and race conditions.")
    else:
        st.success("✔ Low Structural Risk Event.")


