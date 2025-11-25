import os
import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Predicción de PM2.5",
    page_icon="🌫️",
    layout="centered",
)


# =========================
# 1. Descarga y carga de artefactos
# =========================

# URL del modelo desde secrets de Streamlit (NO visible en GitHub)
MODEL_URL = st.secrets["private"]["MODEL_URL"]
MODEL_PATH = "rf_model.pkl"


def download_model_from_drive():
    """Descarga el modelo desde Google Drive si no existe localmente."""
    if os.path.exists(MODEL_PATH):
        return

    st.write("Descargando modelo desde almacenamiento seguro... (solo la primera vez)")
    import gdown  # se instala vía requirements.txt

    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


@st.cache_resource
def load_artifacts():
    # 1) Descargar modelo si hace falta
    download_model_from_drive()

    # 2) Cargar modelo, scaler y lista de columnas
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, feature_cols


model, scaler, feature_cols = load_artifacts()


# =========================
# Título y descripción
# =========================
st.title("🌫️ Predicción de PM2.5 en Beijing")
st.write(
    "Aplicación demo del **Proyecto de Aprendizaje de Máquina** "
    "– Maestría en Inteligencia Artificial (PUJ)."
)
st.markdown(
    "Ajusta las condiciones meteorológicas y temporales para estimar la "
    "concentración de **PM2.5 (µg/m³)**."
)


# =========================
# 2. Inputs del usuario
# =========================
st.subheader("Condiciones de entrada")

# --- Mes, día y hora (sin mostrar año) ---
col_time1, col_time2, col_time3 = st.columns(3)

month_names = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

with col_time1:
    month = st.selectbox(
        "Mes",
        options=list(range(1, 13)),
        format_func=lambda m: month_names[m],
        index=0,
    )

with col_time2:
    day = st.slider("Día del mes", 1, 31, 15, 1)

with col_time3:
    hour = st.slider("Hora del día", 0, 23, 8, 1)

# Clima
col_clima1, col_clima2 = st.columns(2)
with col_clima1:
    temp = st.slider("Temperatura (°C)", -30.0, 45.0, 10.0, 0.5)
    pres = st.slider("Presión (hPa)", 980.0, 1040.0, 1010.0, 0.5)
with col_clima2:
    wind = st.slider("Velocidad del viento (m/s)", 0.0, 15.0, 2.0, 0.1)
    rain = st.slider("Lluvia (mm/h)", 0.0, 20.0, 0.0, 0.1)

hum = st.slider("Humedad relativa aproximada (%)", 0, 100, 50, 1)

pm_prev = st.slider(
    "PM2.5 promedio últimas horas (µg/m³) (aprox.)",
    0.0, 250.0, 20.0, 1.0,
)

# Estación (one-hot)
station_names = [
    "Referencia (otra estación)",  # categoría base (sin dummy)
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]
station = st.selectbox("Estación de monitoreo", station_names)

# Dirección del viento (one-hot)
wd_names = [
    "Referencia (otra dirección)",  # categoría base (sin dummy)
    "ENE",
    "ESE",
    "N",
    "NE",
    "NNE",
    "NNW",
    "NW",
    "S",
    "SE",
    "SSE",
    "SSW",
    "SW",
    "W",
    "WNW",
    "WSW",
]
wd = st.selectbox("Dirección del viento", wd_names)

st.markdown("---")


# =========================
# 3. Construcción del vector de features
# =========================
def build_feature_vector():
    """
    Construye un DataFrame con las 43 columnas exactas que necesita el modelo.
    - Usa valores ingresados por el usuario.
    - Lags y medias móviles de PM se rellenan con pm_prev.
    - Estación y dirección del viento se codifican como one-hot.
    - El año se fija a un valor de referencia (2016) dentro del rango del dataset.
    """
    data = {col: 0.0 for col in feature_cols}

    # ---- Variables temporales ----
    data["year"] = 2016  # año de referencia
    data["month"] = month
    data["day"] = day
    data["hour"] = hour

    # ---- Variables meteorológicas ----
    data["TEMP"] = temp
    data["PRES"] = pres
    data["RAIN"] = rain
    data["WSPM"] = wind

    # Aproximación simple para DEWP usando temperatura y humedad
    if "DEWP" in data:
        data["DEWP"] = temp - (100 - hum) / 5.0

    # ---- Lags y medias móviles de PM2.5 ----
    lag_cols = ["PM_lag_1", "PM_lag_3", "PM_lag_6", "PM_lag_12", "PM_lag_24"]
    ma_cols = ["PM_ma_3", "PM_ma_12", "PM_ma_24"]

    for col in lag_cols + ma_cols:
        if col in data:
            data[col] = pm_prev

    # ---- One-hot de estación ----
    if station != "Referencia (otra estación)":
        station_col = f"station_{station}"
        if station_col in data:
            data[station_col] = 1.0

    # ---- One-hot de dirección del viento ----
    if wd != "Referencia (otra dirección)":
        wd_col = f"wd_{wd}"
        if wd_col in data:
            data[wd_col] = 1.0

    row = pd.DataFrame([[data[col] for col in feature_cols]], columns=feature_cols)
    return row


# =========================
# 4. Predicción y visualización
# =========================
if st.button("Predecir PM2.5"):
    X = build_feature_vector()
    X_scaled = scaler.transform(X)
    y_pred = float(model.predict(X_scaled)[0])

    st.subheader(f"PM2.5 estimado: {y_pred:.2f} µg/m³")

    if y_pred <= 15:
        nivel = "Buena"
        color = "🟢"
    elif y_pred <= 35:
        nivel = "Moderada"
        color = "🟡"
    elif y_pred <= 55:
        nivel = "Dañina para grupos sensibles"
        color = "🟠"
    else:
        nivel = "Dañina para la salud"
        color = "🔴"

    st.write(f"Nivel de calidad del aire: {color} **{nivel}**")

    st.progress(min(y_pred / 200.0, 1.0))
