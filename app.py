# =========================
# OPERATIONAL RISK ANALYSIS
# =========================

st.subheader("🧠 Análisis Operativo del Evento")

# Clasificación de riesgo
if global_risk_prob > 0.7:
    risk_level = "🔴 ALTO"
elif global_risk_prob > 0.4:
    risk_level = "🟠 MEDIO"
else:
    risk_level = "🟢 BAJO"

# Clasificación de saturación
if total_participants > 1500:
    saturation = "🔴 SATURACIÓN ALTA"
elif total_participants > 800:
    saturation = "🟠 SATURACIÓN MEDIA"
else:
    saturation = "🟢 SATURACIÓN BAJA"

# Reglas operativas
hydration_points = int(total_distance / 5)
medical_teams = max(1, int(total_participants / 500))

wave_start_needed = total_participants > 1200
reinforced_medical = global_predicted_dnf > 0.2

# Mostrar resultados tipo dashboard
col1, col2 = st.columns(2)

with col1:
    st.metric("Nivel de Riesgo Global", risk_level)
    st.metric("Nivel de Saturación", saturation)

with col2:
    st.metric("Puntos de Hidratación Recomendados", hydration_points)
    st.metric("Equipos Médicos Recomendados", medical_teams)

st.markdown("---")

st.subheader("📋 Recomendaciones Operativas Automáticas")

recommendations = []

if hydration_points > 0:
    recommendations.append(f"• Instalar al menos {hydration_points} puntos de hidratación (1 cada 5 km).")

if wave_start_needed:
    recommendations.append("• Implementar salidas por bloques (wave start) para reducir congestión inicial.")

if reinforced_medical:
    recommendations.append("• Reforzar presencia médica y equipos de rescate en zonas técnicas.")

if global_risk_prob > 0.6:
    recommendations.append("• Considerar ampliar personal de control y voluntarios en tramos críticos.")

if total_participants > 2000:
    recommendations.append("• Evaluar ampliación de zonas de meta y recuperación post-carrera.")

if not recommendations:
    recommendations.append("• El evento presenta condiciones operativas estables según el modelo.")

for rec in recommendations:
    st.markdown(rec)

st.markdown("---")

st.subheader("📝 Resumen Ejecutivo")

st.info(f"""
El evento simulado presenta un nivel de riesgo {risk_level} con una probabilidad estimada de evento crítico del {round(global_risk_prob*100,1)}%.

La tasa estimada de abandono es del {round(global_predicted_dnf*100,1)}%, lo que sugiere un nivel de exigencia técnica acorde con los parámetros ingresados.

Desde una perspectiva operativa, se recomienda implementar {hydration_points} puntos de hidratación y al menos {medical_teams} equipos médicos distribuidos estratégicamente a lo largo del recorrido.

Este análisis integra variables estructurales del evento y proporciona una herramienta de planificación preventiva orientada a seguridad del atleta y sostenibilidad organizativa.
""")


