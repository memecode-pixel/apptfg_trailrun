import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODELS
# =========================

rf_model = joblib.load("dnf_rf_model.joblib")
log_model = joblib.load("high_dnf_log_model.joblib")

st.title("Trail Event Capacity & Risk Simulator")

st.markdown("---")

# =========================
# GLOBAL EVENT PARAMETERS
# =========================

st.header("Parametros Generales del Evento")

year = st.number_input("Año del evento", min_value=2024, max_value=2035, value=2025)
price = st.number_input("Precio inscripción ($)", min_value=1.0, value=60.0)
fixed_cost = st.number_input("Costo fijo estimado ($)", min_value=0.0, value=15000.0)

st.markdown("---")

# =========================
# ROUTES CONFIGURATION
# =========================

st.header("Configuracion de Rutas")

num_routes = st.number_input("Numero de rutas", min_value=1, max_value=5, value=1)

routes_data = []

for i in range(num_routes):
    st.subheader(f"Ruta {i+1}")
    
    distance = st.number_input(f"Distancia ruta {i+1} (km)", min_value=1.0, value=30.0, key=f"d{i}")
    elevation = st.number_input(f"Desnivel ruta {i+1} (m)", min_value=0.0, value=1500.0, key=f"e{i}")
    participants = st.number_input(f"Participantes ruta {i+1}", min_value=1, value=300, key=f"p{i}")
    start_time = st.time_input(f"Hora salida ruta {i+1}", key=f"t{i}")

    elevation_per_km = elevation / distance if distance > 0 else 0

    routes_data.append({
        "Distance": distance,
        "Elevation Gain": elevation,
        "elevation_per_km": elevation_per_km,
        "N Participants": participants,
        "Year": year
    })

df_routes = pd.DataFrame(routes_data)

# =========================
# MODEL PREDICTIONS
# =========================

dnf_pred = rf_model.predict(df_routes)
risk_prob = log_model.predict_proba(df_routes)[:, 1]

df_routes["Predicted_DNF"] = dnf_pred
df_routes["High_Risk_Prob"] = risk_prob

# =========================
# GLOBAL METRICS
# =========================

total_participants = df_routes["N Participants"].sum()
total_revenue = total_participants * price
profit = total_revenue - fixed_cost
avg_dnf = df_routes["Predicted_DNF"].mean()
avg_risk = df_routes["High_Risk_Prob"].mean()

# Saturation logic
capacity_limit = 1200
saturation_ratio = total_participants / capacity_limit

if saturation_ratio > 1:
    saturation_status = "SOBRESATURADO"
elif saturation_ratio > 0.8:
    saturation_status = "CERCANO A SATURACION"
else:
    saturation_status = "CAPACIDAD ADECUADA"

# =========================
# DASHBOARD METRICS
# =========================

st.markdown("---")
st.header("Resumen Ejecutivo")

col1, col2, col3 = st.columns(3)

col1.metric("Total Participantes", total_participants)
col2.metric("Beneficio Estimado ($)", round(profit, 2))
col3.metric("Riesgo Promedio DNF", round(avg_dnf, 3))

col4, col5 = st.columns(2)

col4.metric("Probabilidad Alta Riesgo", round(avg_risk, 3))
col5.metric("Estado de Capacidad", saturation_status)

# =========================
# VISUALIZATION
# =========================

st.markdown("---")
st.header("Visualizacion de Riesgo por Ruta")

st.bar_chart(df_routes[["Predicted_DNF", "High_Risk_Prob"]])

# =========================
# OPERATIONAL RECOMMENDATIONS
# =========================

st.markdown("---")
st.header("Recomendaciones Operativas")

for idx, row in df_routes.iterrows():
    st.subheader(f"Ruta {idx+1}")

    if row["High_Risk_Prob"] > 0.7:
        st.warning("Riesgo alto detectado. Se recomienda:")
        st.write("- Punto de hidratacion cada 5 km")
        st.write("- Personal medico adicional")
        st.write("- Mayor señalizacion tecnica")
    elif row["High_Risk_Prob"] > 0.4:
        st.info("Riesgo medio. Recomendado:")
        st.write("- Punto de hidratacion cada 7-8 km")
        st.write("- Control de tiempos intermedios")
    else:
        st.success("Riesgo bajo. Operacion estandar suficiente.")

# =========================
# CAPACITY ALERT
# =========================

if saturation_ratio > 1:
    st.error("El evento supera la capacidad recomendada.")
elif saturation_ratio > 0.8:
    st.warning("El evento esta cercano al limite de capacidad.")
else:
    st.success("El evento esta dentro de parametros de capacidad.")






