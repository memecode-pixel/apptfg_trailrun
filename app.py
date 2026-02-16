import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rf_model = joblib.load("dnf_rf_model.joblib")
log_model = joblib.load("high_dnf_log_model.joblib")

st.title("Trail Event Capacity & Risk Simulator")

st.markdown("### Parámetros del evento")

distance = st.number_input("Distancia (km)", min_value=1.0)
elevation = st.number_input("Desnivel acumulado (m)", min_value=0.0)
year = st.number_input("Año del evento", min_value=2000, max_value=2035)
fee = st.number_input("Cuota inscripción (€)", min_value=0.0)
fixed_cost = st.number_input("Costo fijo estimado (€)", min_value=0.0)
max_participants = st.number_input("Máximo participantes a simular", min_value=50, value=1000)

if st.button("Simular Escenarios"):

    elevation_per_km = elevation / distance if distance > 0 else 0
    participants_range = np.arange(50, max_participants + 1, 50)

    results = []

    for p in participants_range:
        input_data = pd.DataFrame({
            'Distance': [distance],
            'Elevation Gain': [elevation],
            'elevation_per_km': [elevation_per_km],
            'N Participants': [p],
            'Year': [year]
        })

        predicted_dnf = rf_model.predict(input_data)[0]
        prob_high = log_model.predict_proba(input_data)[0][1]

        revenue = p * fee
        profit = revenue - fixed_cost

        results.append({
            "Participants": p,
            "Predicted_DNF": predicted_dnf,
            "High_Risk_Prob": prob_high,
            "Revenue": revenue,
            "Profit": profit
        })

    results_df = pd.DataFrame(results)

    st.markdown("---")
    st.subheader("Resultados de simulación")

    st.dataframe(results_df)

    saturation = results_df[results_df["High_Risk_Prob"] > 0.6]

    if not saturation.empty:
        recommended_limit = saturation["Participants"].iloc[0]
        st.warning(f"Punto estimado de saturación: ~{recommended_limit} participantes")
    else:
        st.success("No se detecta saturación dentro del rango simulado")

    fig, ax1 = plt.subplots()

    ax1.plot(results_df["Participants"], results_df["Predicted_DNF"])
    ax1.set_xlabel("Participantes")
    ax1.set_ylabel("DNF estimado")

    ax2 = ax1.twinx()
    ax2.plot(results_df["Participants"], results_df["Profit"], linestyle="--")
    ax2.set_ylabel("Beneficio estimado (€)")

    st.pyplot(fig)
