import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("Trail Event Capacity & Risk Simulator")

st.subheader("Configuración del Evento")

year = st.number_input("Año del evento", 2024, 2035, 2025)
entry_fee = st.number_input("Cuota inscripción ($)", 0.0, 1000.0, 60.0)
fixed_cost = st.number_input("Costo fijo estimado ($)", 0.0, 1000000.0, 15000.0)

st.markdown("---")
st.subheader("Configuración de Rutas")

num_routes = st.number_input("Cantidad de rutas", 1, 5, 1)

routes = []

for i in range(num_routes):
    st.markdown(f"### Ruta {i+1}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        distance = st.number_input(f"Distancia km (Ruta {i+1})", 1.0, 200.0, 50.0, key=f"d{i}")
    with col2:
        elevation = st.number_input(f"Desnivel m (Ruta {i+1})", 0.0, 15000.0, 2000.0, key=f"e{i}")
    with col3:
        participants = st.number_input(f"Participantes (Ruta {i+1})", 0, 10000, 500, key=f"p{i}")
    
    elev_per_km = elevation / distance if distance > 0 else 0
    
    routes.append({
        "Distance": distance,
        "Elevation Gain": elevation,
        "elevation_per_km": elev_per_km,
        "N Participants": participants,
        "Year": year
    })

if st.button("Simular Evento"):

    df_routes = pd.DataFrame(routes)

    # --- Predicciones ---
    dnf_pred = rf_model.predict(df_routes)
    high_risk_prob = log_model.predict_proba(df_routes)[:,1]

    df_routes["Predicted_DNF"] = dnf_pred
    df_routes["High_Risk_Prob"] = high_risk_prob

    total_participants = df_routes["N Participants"].sum()
    revenue = total_participants * entry_fee
    profit = revenue - fixed_cost

    global_risk = high_risk_prob.mean()
    avg_dnf = dnf_pred.mean()

    # Saturación simple (modelo heurístico)
    capacity_estimate = int(1000 - (avg_dnf * 500))
    saturation_ratio = total_participants / capacity_estimate if capacity_estimate > 0 else 0

    # ======================
    # DASHBOARD METRICS
    # ======================

    st.markdown("---")
    st.subheader("Resumen Ejecutivo")

    col1, col2, col3 = st.columns(3)

    col1.metric("Participantes Totales", f"{total_participants:,}")
    col2.metric("Ingreso Estimado ($)", f"${revenue:,.0f}")
    col3.metric("Beneficio Estimado ($)", f"${profit:,.0f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("DNF Promedio Estimado", f"{avg_dnf:.2%}")
    col5.metric("Probabilidad Riesgo Alto", f"{global_risk:.2%}")
    col6.metric("Capacidad Recomendada", f"{capacity_estimate:,}")

    # ======================
    # RIESGO
    # ======================

    st.markdown("---")
    st.subheader("Evaluación de Riesgo")

    if global_risk > 0.7:
        st.error("Riesgo Global ALTO")
    elif global_risk > 0.4:
        st.warning("Riesgo Global MEDIO")
    else:
        st.success("Riesgo Global BAJO")

    if saturation_ratio > 1:
        st.error("Evento Saturado - Reduce participantes o aumenta soporte logístico")
    elif saturation_ratio > 0.8:
        st.warning("Evento cercano a saturación")
    else:
        st.success("Capacidad dentro de rango seguro")

    # ======================
    # RECOMENDACIONES OPERATIVAS
    # ======================

    st.markdown("---")
    st.subheader("Recomendaciones Operativas")

    hydration_points = int(df_routes["Distance"].mean() / 5)
    medical_points = int(df_routes["Distance"].mean() / 15)

    st.write(f"- Centros de hidratación recomendados: {hydration_points}")
    st.write(f"- Puntos médicos recomendados: {medical_points}")
    st.write(f"- Staff mínimo sugerido: {int(total_participants / 50)} personas")

    # ======================
    # GRÁFICO EXPLICATIVO
    # ======================

    st.markdown("---")
    st.subheader("Visualización de Riesgo vs Participantes")

    fig, ax1 = plt.subplots()

    ax1.bar(range(len(df_routes)), df_routes["N Participants"])
    ax1.set_ylabel("Participantes")

    ax2 = ax1.twinx()
    ax2.plot(df_routes["High_Risk_Prob"], linestyle="--")
    ax2.set_ylabel("Probabilidad Riesgo Alto")

    st.pyplot(fig)

    # Tabla detallada
    st.markdown("---")
    st.subheader("Detalle por Ruta")
    st.dataframe(df_routes)






