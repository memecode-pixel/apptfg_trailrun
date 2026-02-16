import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Trail Event Intelligence", layout="wide")

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    rf_model = joblib.load("dnf_rf_model.joblib")
    log_model = joblib.load("high_dnf_log_model.joblib")
    return rf_model, log_model

rf_model, log_model = load_models()

# =========================
# HEADER
# =========================

st.title("🏔 Trail Event Intelligence Platform")
st.markdown("Simulate capacity, structural risk and profitability across multiple race routes.")

st.divider()

# =========================
# GLOBAL EVENT PARAMETERS
# =========================

st.subheader("Global Event Settings")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Event Year", min_value=2014, max_value=2035, value=2025)

with col2:
    fixed_cost = st.number_input("Estimated Fixed Cost ($)", min_value=0.0, value=20000.0)

st.divider()

# =========================
# NUMBER OF ROUTES
# =========================

n_routes = st.number_input("Number of Routes", min_value=1, max_value=5, value=3)

routes_data = []

st.subheader("Route Configuration")

for i in range(int(n_routes)):

    st.markdown(f"### Route {i+1}")

    colA, colB, colC = st.columns(3)

    with colA:
        distance = st.number_input(f"Distance (km) - Route {i+1}", min_value=1.0, value=20.0, key=f"d{i}")
        elevation_gain = st.number_input(f"Elevation Gain (m) - Route {i+1}", min_value=0.0, value=800.0, key=f"e{i}")

    with colB:
        participants = st.number_input(f"Participants - Route {i+1}", min_value=1, value=300, key=f"p{i}")
        fee = st.number_input(f"Registration Fee ($) - Route {i+1}", min_value=0.0, value=80.0, key=f"f{i}")

    with colC:
        start_hour = st.number_input(f"Start Hour (24h) - Route {i+1}", min_value=0, max_value=23, value=6+i, key=f"s{i}")

    routes_data.append({
        "route": i+1,
        "distance": distance,
        "elevation_gain": elevation_gain,
        "participants": participants,
        "fee": fee,
        "start_hour": start_hour
    })

st.divider()

# =========================
# SIMULATION
# =========================

if st.button("🚀 Run Event Simulation"):

    results = []

    total_participants = 0
    total_revenue = 0

    for route in routes_data:

        elevation_per_km = route["elevation_gain"] / route["distance"]

        features = pd.DataFrame([{
            "Distance": route["distance"],
            "Elevation Gain": route["elevation_gain"],
            "elevation_per_km": elevation_per_km,
            "N Participants": route["participants"],
            "Year": year
        }])

        predicted_dnf = rf_model.predict(features)[0]
        high_risk_prob = log_model.predict_proba(features)[0][1]

        revenue = route["participants"] * route["fee"]

        results.append({
            "Route": route["route"],
            "Participants": route["participants"],
            "Predicted DNF": predicted_dnf,
            "High Risk Prob": high_risk_prob,
            "Revenue ($)": revenue
        })

        total_participants += route["participants"]
        total_revenue += revenue

    df_results = pd.DataFrame(results)

    total_profit = total_revenue - fixed_cost
    weighted_risk = np.average(df_results["High Risk Prob"], weights=df_results["Participants"])
    weighted_dnf = np.average(df_results["Predicted DNF"], weights=df_results["Participants"])

    # =========================
    # DASHBOARD METRICS
    # =========================

    st.subheader("📊 Event Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Participants", f"{total_participants:,}")
    col2.metric("Total Revenue ($)", f"${total_revenue:,.0f}")
    col3.metric("Estimated Profit ($)", f"${total_profit:,.0f}")
    col4.metric("Weighted High Risk", f"{weighted_risk:.2%}")

    st.divider()

    st.subheader("🏁 Route-Level Results")
    st.dataframe(df_results, use_container_width=True)

    st.divider()

    # =========================
    # RISK STATUS
    # =========================

    if weighted_risk > 0.7:
        st.error("⚠ High Structural Risk Event – Consider reducing capacity or increasing safety controls.")
    elif weighted_risk > 0.4:
        st.warning("⚠ Moderate Risk – Monitor density and route logistics.")
    else:
        st.success("✔ Event structurally balanced.")


