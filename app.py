import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Trail Event Risk Simulator", layout="wide")

# =========================
# LOAD MODELS
# =========================

rf_model = joblib.load("dnf_rf_model.joblib")
log_model = joblib.load("high_dnf_log_model.joblib")

# =========================
# TITLE
# =========================

st.title("Trail Event Capacity & Risk Simulator")

# =========================
# INPUTS
# =========================

st.sidebar.header("Parametros del Evento")

distance = st.sidebar.number_input("Distancia (km)", min_value=1.0, value=50.0)
elevation = st.sidebar.number_input("Desnivel acumulado (m)", min_value=0.0, value=2000.0)
participants = st.sidebar.number_input("Numero de participantes", min_value=1, value=500)
year = st.sidebar.number_input("Año del evento", min_value=2014, max_value=2035, value=2025)
entry_fee = st.sidebar.number_input("Inscripcion por participante ($)", min_value=0.0, value=80.0)
fixed_cost = st.sidebar.number_input("Costo fijo estimado ($)", min_value=0.0, value=20000.0)

# =========================
# FEATURE ENGINEERING
# =========================

elevation_per_km = elevation / distance

input_df = pd.DataFrame([{
    "Distance": distance,
    "Elevation Gain": elevation,
    "elevation_per_km": elevation_per_km,
    "N Participants": participants,
    "Year": year
}])

# =========================
# MODEL PREDICTIONS
# =========================

predicted_dnf = rf_model.predict(input_df)[0]
risk_prob = log_model.predict_proba(input_df)[0][1]

revenue = participants * entry_fee
profit = revenue - fixed_cost

# =========================
# METRICS DASHBOARD
# =========================

col1, col2, col3 = st.columns(3)

col1.metric("DNF Estimado", f"{predicted_dnf:.2%}")
col2.metric("Probabilidad Alto Riesgo", f"{risk_prob:.2%}")
col3.metric("Beneficio Estimado", f"${profit:,.0f}")

# =========================
# OPERATIONAL RISK ANALYSIS
# =========================

st.subheader("Analisis Operativo del Evento")

if risk_prob > 0.7:
    risk_level = "ALTO"
elif risk_prob > 0.4:
    risk_level = "MEDIO"
else:
    risk_level = "BAJO"

st.write("Nivel de riesgo:", risk_level)

# =========================
# RECOMMENDATIONS ENGINE
# =========================

st.subheader("Recomendaciones Operativas")

hydration_points = int(distance / 5)
medical_points = 1 if risk_level == "BAJO" else 2 if risk_level == "MEDIO" else 3

if participants > 1000:
    wave_starts = 3
elif participants > 500:
    wave_starts = 2
else:
    wave_starts = 1

if risk_level == "ALTO":
    recommendation_text = """
    - Aumentar puntos de hidratacion cada 5 km.
    - Implementar salidas escalonadas.
    - Reforzar equipo medico en ruta.
    - Limitar capacidad maxima.
    """
elif risk_level == "MEDIO":
    recommendation_text = """
    - Mantener puntos de hidratacion cada 5 km.
    - Evaluar salida en 2 olas.
    - Supervisar zonas tecnicas.
    """
else:
    recommendation_text = """
    - Operacion estandar.
    - Mantener monitoreo basico.
    """

st.write("Puntos de hidratacion recomendados:", hydration_points)
st.write("Puestos medicos recomendados:", medical_points)
st.write("Numero de salidas sugeridas:", wave_starts)
st.markdown(recommendation_text)

# =========================
# SATURATION SIMULATION
# =========================

st.subheader("Simulacion de Saturacion")

max_sim = st.slider("Simular hasta cuantos participantes", 100, 5000, 2000)

simulation_results = []

for p in range(100, max_sim, 100):
    temp_df = pd.DataFrame([{
        "Distance": distance,
        "Elevation Gain": elevation,
        "elevation_per_km": elevation_per_km,
        "N Participants": p,
        "Year": year
    }])
    
    temp_dnf = rf_model.predict(temp_df)[0]
    temp_risk = log_model.predict_proba(temp_df)[0][1]
    temp_profit = p * entry_fee - fixed_cost
    
    simulation_results.append({
        "Participants": p,
        "Predicted_DNF": temp_dnf,
        "Risk_Prob": temp_risk,
        "Profit": temp_profit
    })

sim_df = pd.DataFrame(simulation_results)

st.line_chart(sim_df.set_index("Participants")[["Risk_Prob", "Predicted_DNF"]])
st.line_chart(sim_df.set_index("Participants")["Profit"])

# Punto estimado de saturacion
sat_point = sim_df[sim_df["Risk_Prob"] > 0.7]

if not sat_point.empty:
    st.warning(f"Punto estimado de saturacion operacional: ~{sat_point.iloc[0]['Participants']} participantes")
else:
    st.success("No se detecta saturacion en el rango simulado.")




