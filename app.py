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
# EVENT FIXED COST
# =========================

st.header("Parametros Generales")

year = st.number_input("Año del evento", min_value=2024, max_value=2035, value=2025)
fixed_cost = st.number_input("Costo fijo total del evento ($)", min_value=0.0, value=15000.0)

st.markdown("---")

# =========================
# ROUTES CONFIGURATION
# =========================

st.header("Configuracion de Rutas")

num_routes = st.number_input("Numero de rutas", min_value=1, max_value=5, value=1)

routes_data = []

for i in range(num_routes):
    st.subheader(f"Ruta {i+1}")
    
    distance = st.number_input(f"Distancia (km)", min_value=1.0, value=30.0, key=f"d{i}")
    elevation = st.number_input(f"Desnivel (m)", min_value=0.0, value=1500.0, key=f"e{i}")
    participants = st.number_input(f"Participantes", min_value=1, value=300, key=f"p{i}")
    price = st.number_input(f"Precio inscripción ($)", min_value=1.0, value=60.0, key=f"price{i}")
    variable_cost = st.number_input(f"Costo variable por participante ($)", min_value=0.0, value=15.0, key=f"vc{i}")

    elevation_per_km = elevation / distance if distance > 0 else 0

    revenue = participants * price
    total_variable_cost = participants * variable_cost
    route_profit = revenue - total_variable_cost

    routes_data.append({
        "Distance": distance,
        "Elevation Gain": elevation,
        "elevation_per_km": elevation_per_km,
        "N Participants": participants,
        "Year": year,
        "Revenue": revenue,
        "Variable_Cost": total_variable_cost,
        "Route_Profit": route_profit
    })

df_routes = pd.DataFrame(routes_data)

# =========================
# MODEL PREDICTIONS
# =========================

model_features = df_routes[[
    "Distance",
    "Elevation Gain",
    "elevation_per_km",
    "N Participants",
    "Year"
]]

dnf_pred = rf_model.predict(model_features)
risk_prob = log_model.predict_proba(model_features)[:, 1]

df_routes["Predicted_DNF"] = dnf_pred
df_routes["High_Risk_Prob"] = risk_prob

# =========================
# GLOBAL METRICS
# =========================

total_participants = df_routes["N Participants"].sum()
total_revenue = df_routes["Revenue"].sum()
total_variable_cost = df_routes["Variable_Cost"].sum()
total_profit = total_revenue - total_variable_cost - fixed_cost
avg_risk = df_routes["High_Risk_Prob"].mean()

# Capacidad sugerida simple
capacity_limit = 1200
saturation_ratio = total_participants / capacity_limit

if saturation_ratio > 1:
    saturation_status = "SOBRESATURADO"
elif saturation_ratio > 0.8:
    saturation_status = "CERCANO AL LIMITE"
else:
    saturation_status = "CAPACIDAD ADECUADA"

# =========================
# DASHBOARD METRICS
# =========================

st.markdown("---")
st.header("Resumen Ejecutivo")

col1, col2, col3 = st.columns(3)

col1.metric("Total Participantes", total_participants)
col2.metric("Ingreso Total ($)", round(total_revenue, 2))
col3.metric("Beneficio Total ($)", round(total_profit, 2))

col4, col5 = st.columns(2)

col4.metric("Riesgo Promedio Alto DNF", round(avg_risk, 3))
col5.metric("Estado de Capacidad", saturation_status)

# =========================
# ROUTE BREAKDOWN TABLE
# =========================

st.markdown("---")
st.header("Detalle por Ruta")

st.dataframe(df_routes[[
    "N Participants",
    "Revenue",
    "Variable_Cost",
    "Route_Profit",
    "Predicted_DNF",
    "High_Risk_Prob"
]])

# =========================
# VISUAL RISK CHART
# =========================

st.markdown("---")
st.header("Comparacion de Riesgo por Ruta")

st.bar_chart(df_routes[["Predicted_DNF", "High_Risk_Prob"]])

# =========================
# OPERATIONAL RECOMMENDATIONS
# =========================

st.markdown("---")
st.header("Recomendaciones Operativas")

for idx, row in df_routes.iterrows():
    st.subheader(f"Ruta {idx+1}")

    if row["High_Risk_Prob"] > 0.7:
        st.warning("Riesgo alto.")
        st.write("- Hidratacion cada 5 km")
        st.write("- Refuerzo medico")
        st.write("- Mayor control tecnico")
    elif row["High_Risk_Prob"] > 0.4:
        st.info("Riesgo medio.")
        st.write("- Hidratacion cada 7 km")
        st.write("- Monitoreo intermedio")
    else:
        st.success("Riesgo bajo. Operacion estandar suficiente.")

# =========================
# CAPACITY ALERT
# =========================

st.markdown("---")

if saturation_ratio > 1:
    st.error("El evento supera la capacidad recomendada.")
elif saturation_ratio > 0.8:
    st.warning("El evento esta cercano al limite operativo.")
else:
    st.success("Evento dentro de capacidad operativa.") 

# =========================
# GLOBAL METRICS
# =========================

total_participants = df_routes["N Participants"].sum()
total_revenue = df_routes["Revenue"].sum()
total_variable_cost = df_routes["Variable_Cost"].sum()
total_profit = total_revenue - total_variable_cost - fixed_cost
avg_risk = df_routes["High_Risk_Prob"].mean()

# Capacidad base configurable
capacity_limit = 1200
saturation_ratio = total_participants / capacity_limit

# Break-even
if total_participants > 0:
    break_even_price = (fixed_cost / total_participants) + (
        df_routes["Variable_Cost"].sum() / total_participants
    )
else:
    break_even_price = 0

# =========================
# NIVEL RECOMENDADO POR RUTA
# =========================

recommended_participants = []

for idx, row in df_routes.iterrows():
    if row["High_Risk_Prob"] > 0.7:
        recommended = int(row["N Participants"] * 0.8)
    elif row["High_Risk_Prob"] > 0.4:
        recommended = int(row["N Participants"] * 0.9)
    else:
        recommended = int(row["N Participants"] * 1.05)
    recommended_participants.append(recommended)

df_routes["Recommended_Participants"] = recommended_participants

# =========================
# DASHBOARD METRICS
# =========================

st.markdown("---")
st.header("Resumen Ejecutivo")

col1, col2, col3 = st.columns(3)
col1.metric("Total Participantes", total_participants)
col2.metric("Ingreso Total ($)", round(total_revenue, 2))
col3.metric("Beneficio Total ($)", round(total_profit, 2))

col4, col5, col6 = st.columns(3)
col4.metric("Riesgo Promedio", round(avg_risk, 3))
col5.metric("Break-even Precio ($)", round(break_even_price, 2))
col6.metric("Capacidad Utilizada (%)", round(saturation_ratio * 100, 1))

# =========================
# TABLA DETALLE
# =========================

st.markdown("---")
st.header("Detalle por Ruta")

st.dataframe(df_routes[[
    "N Participants",
    "Recommended_Participants",
    "Revenue",
    "Route_Profit",
    "Predicted_DNF",
    "High_Risk_Prob"
]])

# =========================
# GRAFICO PROFESIONAL
# =========================

st.markdown("---")
st.header("Riesgo vs Beneficio por Ruta")

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# Beneficio
ax1.bar(
    range(len(df_routes)),
    df_routes["Route_Profit"],
    alpha=0.6,
    label="Beneficio"
)

ax1.set_ylabel("Beneficio ($)")

# Riesgo en eje secundario
ax2 = ax1.twinx()
ax2.plot(
    range(len(df_routes)),
    df_routes["High_Risk_Prob"],
    marker="o",
    linewidth=3,
    label="Riesgo DNF"
)

ax2.set_ylabel("Probabilidad Alto DNF")

plt.title("Balance Riesgo - Rentabilidad")
plt.xticks(range(len(df_routes)), [f"Ruta {i+1}" for i in range(len(df_routes))])

st.pyplot(fig)

# =========================
# ALERTAS DE CAPACIDAD
# =========================

st.markdown("---")
st.header("Estado Operativo")

if saturation_ratio > 1:
    st.error("Evento sobresaturado. Se recomienda reducir cupos.")
elif saturation_ratio > 0.8:
    st.warning("Evento cercano al limite operativo.")
else:
    st.success("Capacidad operativa adecuada.")

# =========================
# RECOMENDACIONES OPERATIVAS
# =========================

st.markdown("---")
st.header("Recomendaciones")

for idx, row in df_routes.iterrows():
    st.subheader(f"Ruta {idx+1}")

    if row["High_Risk_Prob"] > 0.7:
        st.warning("Riesgo alto.")
        st.write("• Hidratacion cada 5 km")
        st.write("• Centro medico intermedio")
        st.write("• Mayor control tecnico")
    elif row["High_Risk_Prob"] > 0.4:
        st.info("Riesgo medio.")
        st.write("• Hidratacion cada 7 km")
        st.write("• Monitoreo adicional")
    else:
        st.success("Riesgo bajo. Operacion estandar suficiente.")






