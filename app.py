import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Trail Event Simulator", layout="wide")

# -----------------------------
# Cargar dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("TU_DATASET.csv")  # <-- pon aquí tu dataset real
    df['DNF_rate'] = df['N DNF'] / df['N Participants']
    df['elevation_per_km'] = df['Elevation Gain'] / df['Distance']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

df = load_data()

features = [
    'Distance',
    'Elevation Gain',
    'elevation_per_km',
    'N Participants',
    'Year'
]

# -----------------------------
# Entrenar modelos dentro de la app
# -----------------------------
@st.cache_resource
def train_models(df):

    X = df[features]
    y_reg = df['DNF_rate']

    # Modelo regresión (DNF continuo)
    rf_model = RandomForestRegressor(
        n_estimators=80,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X, y_reg)

    # Modelo clasificación alto riesgo
    threshold = df['DNF_rate'].quantile(0.75)
    df['High_DNF'] = (df['DNF_rate'] >= threshold).astype(int)

    y_clf = df['High_DNF']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    return rf_model, log_model, threshold

rf_model, log_model, threshold = train_models(df)

# -----------------------------
# Simulación
# -----------------------------
st.title("Trail Event Capacity & Risk Simulator")

distance = st.number_input("Distancia (km)", 1.0, 200.0, 50.0)
elevation = st.number_input("Desnivel acumulado (m)", 0.0, 10000.0, 1200.0)
participants = st.number_input("Número de participantes", 10, 10000, 500)
year = st.number_input("Año", 2010, 2035, 2025)

elevation_per_km = elevation / distance

input_data = pd.DataFrame([{
    'Distance': distance,
    'Elevation Gain': elevation,
    'elevation_per_km': elevation_per_km,
    'N Participants': participants,
    'Year': year
}])

# Predicciones
predicted_dnf = rf_model.predict(input_data)[0]
high_risk_prob = log_model.predict_proba(input_data)[0][1]

st.markdown("### 📊 Resultados")

col1, col2 = st.columns(2)

with col1:
    st.metric("DNF estimado", f"{predicted_dnf:.2%}")

with col2:
    st.metric("Probabilidad Alto Riesgo", f"{high_risk_prob:.2%}")

if high_risk_prob > 0.5:
    st.error("⚠️ Evento con alto riesgo estructural")
else:
    st.success("✅ Riesgo estructural controlado")
