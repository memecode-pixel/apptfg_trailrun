import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================
# OPERATIONAL RISK ANALYSIS
# =========================

st.subheader("Analisis Operativo del Evento")

# Clasificación de riesgo
if global_risk_prob > 0.7:
    risk_level = "ALTO"
elif global_risk_prob > 0.4:
    risk_level = "MEDIO"
else:
    risk_level = "BAJO"

# Clasificación de saturación
if total_participants > 1500:
    saturation = "SATURACION ALTA"
elif total_participants > 800:
    saturation = "SATURACION MEDIA"
else:
    saturation = "SATURACION BAJA"

# Reglas operativas
hydration_points = int(total_distance / 5)
medical_teams = max(1, int(total_participants / 500))

wave_start_needed = total_participants > 1200
reinforced_medical = global_predicted_dnf > 0.2

# Mostrar métricas
col1, col2 = st.columns(2)

with col1:
    st.metric("Nivel de Riesgo Global", risk_level)
    st.metric("Nivel de Saturacion", saturation)

with col2:
    st.metric("Puntos de Hidratacion Recomendados", hydration_points)
    st.metric("Equipos Medicos Recomendados", medical_teams)

st.markdown("---")

st.subheader("Recomendaciones Operativas")

recommendations = []

if hydration_points > 0:
    recommendations.append(f"- Instalar al menos {hydration_points} puntos de hidratacion (1 cada 5 km).")

if wave_start_needed:
    recommendations.append("- Implementar salidas por bloques para reducir congestion inicial.")

if reinforced_medical:
    recommendations.append("- Reforzar presencia medica y equipos de rescate en zonas tecnicas.")

if global_risk_prob > 0.6:
    recommendations.append("- Ampliar personal de control y voluntarios en tramos criticos.")

if total_participants > 2000:
    recommendations.append("- Evaluar ampliacion de zonas de meta y recuperacion post-carrera.")

if not recommendations:
    recommendations.append("- El evento presenta condiciones operativas estables segun el modelo.")

for rec in recommendations:
    st.markdown(rec)

st.markdown("---")

st.subheader("Resumen Ejecutivo")

st.info(f"""
El evento presenta un nivel de riesgo {risk_level} con una probabilidad estimada de evento critico del {round(global_risk_prob*100,1)}%.

La tasa estimada de abandono es del {round(global_predicted_dnf*100,1)}%.

Se recomienda implementar {hydration_points} puntos de hidratacion y al menos {medical_teams} equipos medicos distribuidos estrategicamente.

Este analisis integra variables estructurales del evento y proporciona una herramienta de planificacion preventiva orientada a la seguridad del atleta.
""")


