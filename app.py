import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.sidebar.header("Configuracion del Evento")

num_routes = st.sidebar.number_input("Numero de rutas", min_value=1, max_value=5, value=1)

routes_data = []

for i in range(num_routes):
    st.sidebar.markdown(f"---")
    st.sidebar.subheader(f"Ruta {i+1}")
    
    distance = st.sidebar.number_input(f"Distancia Ruta {i+1} (km)", min_value=1.0, value=25.0, key=f"d{i}")
    elevation = st.sidebar.number_input(f"Desnivel Ruta {i+1} (m)", min_value=0.0, value=1000.0, key=f"e{i}")
    participants = st.sidebar.number_input(f"Participantes Ruta {i+1}", min_value=1, value=300, key=f"p{i}")
    start_hour = st.sidebar.number_input(f"Hora inicio Ruta {i+1}", min_value=0, max_value=23, value=6, key=f"h{i}")
    
    elevation_per_km = elevation / distance
    
    input_df = pd.DataFrame([{
        "Distance": distance,
        "Elevation Gain": elevation,
        "elevation_per_km": elevation_per_km,
        "N Participants": participants,
        "Year": year
    }])
    
    predicted_dnf = rf_model.predict(input_df)[0]
    risk_prob = log_model.predict_proba(input_df)[0][1]
    
    routes_data.append({
        "Ruta": f"Ruta {i+1}",
        "Distance": distance,
        "Participants": participants,
        "DNF": predicted_dnf,
        "Risk": risk_prob
    })





