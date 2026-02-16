import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Trail Event Risk Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #111;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Trail Event Capacity & Risk Simulator")

# -----------------------------
# LOAD MODELS
# -----------------------------
rf_model = joblib.load("dnf_rf_model.joblib")
log_model = joblib.load("high_dnf_log_model.joblib")
growth_model = joblib.load("growth_model.joblib")

# -----------------------------
# INPUTS
# -----------------------------
st.sidebar.header("Event Parameters")

distance = st.sidebar.number_input("Distance (km)", 1.0, 300.0, 50.0)
elevation = st.sidebar.number_input("Elevation Gain (m)", 0.0, 15000.0, 2000.0)
year = st.sidebar.number_input("Year", 2024, 2035, 2025)
fee = st.sidebar.number_input("Registration Fee ($)", 10.0, 1000.0, 60.0)
fixed_cost = st.sidebar.number_input("Fixed Cost ($)", 0.0, 500000.0, 20000.0)
max_participants = st.sidebar.number_input("Max Participants to Simulate", 100, 10000, 3000)

simulate = st.sidebar.button("Run Simulation")

# -----------------------------
# SIMULATION
# -----------------------------
if simulate:

    participants_range = np.arange(100, max_participants, 50)

    elevation_per_km = elevation / distance

    df_sim = pd.DataFrame({
        "Distance": distance,
        "Elevation Gain": elevation,
        "elevation_per_km": elevation_per_km,
        "Year": year,
        "N Participants": participants_range
    })

    # Predict DNF
    df_sim["Predicted_DNF"] = rf_model.predict(df_sim)

    # Predict High Risk probability
    df_sim["High_Risk_Prob"] = log_model.predict_proba(df_sim)[:,1]

    # Financial metrics
    df_sim["Revenue"] = df_sim["N Participants"] * fee
    df_sim["Profit"] = df_sim["Revenue"] - fixed_cost
    df_sim["ROI"] = (df_sim["Profit"] / fixed_cost).replace([np.inf, -np.inf], 0)

    # Saturation threshold
    risk_threshold = 0.75
    sat = df_sim[df_sim["High_Risk_Prob"] >= risk_threshold]

    if not sat.empty:
        saturation_point = int(sat.iloc[0]["N Participants"])
    else:
        saturation_point = int(df_sim["N Participants"].max())

    safe_capacity = int(saturation_point * 0.9)
    max_profit = int(df_sim["Profit"].max())
    break_even = int(np.ceil(fixed_cost / fee))

    avg_risk = round(df_sim["High_Risk_Prob"].mean(), 2)
    avg_dnf = round(df_sim["Predicted_DNF"].mean(), 2)

    # -----------------------------
    # DASHBOARD CARDS
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Saturation Point", f"{saturation_point}")
    col2.metric("Safe Capacity", f"{safe_capacity}")
    col3.metric("Max Profit", f"${max_profit:,.0f}")
    col4.metric("Break-even Participants", f"{break_even}")

    st.divider()

    col5, col6, col7 = st.columns(3)

    col5.metric("Average Risk Probability", avg_risk)
    col6.metric("Average Predicted DNF", avg_dnf)
    col7.metric("Expected ROI", f"{round(df_sim['ROI'].max()*100,1)} %")

    st.divider()

    # -----------------------------
    # EXECUTIVE GRAPH
    # -----------------------------
    fig, ax1 = plt.subplots()

    ax1.plot(
        df_sim["N Participants"],
        df_sim["High_Risk_Prob"],
        label="Risk Probability"
    )

    ax1.axvline(saturation_point, linestyle="--")

    ax1.set_xlabel("Participants")
    ax1.set_ylabel("Risk Probability")

    ax2 = ax1.twinx()
    ax2.plot(
        df_sim["N Participants"],
        df_sim["Profit"],
        linestyle="--"
    )
    ax2.set_ylabel("Profit ($)")

    plt.title("Capacity vs Risk & Profit")
    st.pyplot(fig)

    st.divider()

    # -----------------------------
    # EXECUTIVE INTERPRETATION
    # -----------------------------
    st.subheader("Executive Interpretation")

    st.write(
        """
        Revenue increases linearly as participation grows.  
        However, structural abandonment risk also increases beyond a critical threshold.  

        The model identifies a saturation point where operational risk becomes elevated.  
        Maintaining participation below the safe capacity preserves profitability while reducing structural event stress.
        """
    )

