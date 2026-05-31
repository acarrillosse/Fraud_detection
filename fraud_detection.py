"""
fraud_detection.py
------------------
Aplicación web con Streamlit para la detección de fraude en transacciones financieras.
Carga un pipeline de scikit-learn previamente entrenado y permite al usuario introducir
los datos de una transacción para obtener una predicción en tiempo real.

Uso:
    streamlit run fraud_detection.py
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
# El pipeline incluye preprocesamiento (StandardScaler + OneHotEncoder)
# y un clasificador de Regresión Logística entrenado con class_weight="balanced"
# para compensar el fuerte desbalance de clases del dataset (~0,13% de fraudes).
model = joblib.load("fraud_detection_pipeline.pkl")

# ---------------------------------------------------------------------------
# Interfaz de usuario
# ---------------------------------------------------------------------------
st.title("Detección de Fraude en Transacciones")
st.markdown("Introduce los datos de la transacción y pulsa **Predecir** para obtener el resultado.")

st.divider()

# --- Entradas del usuario ---

# Tipo de transacción (variable categórica → OneHotEncoder en el pipeline)
transaction_type = st.selectbox(
    "Tipo de transacción",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
)

# Importe de la transacción
amount = st.number_input("Importe (€)", min_value=0.0, value=1000.0)

# Saldos del emisor antes y después de la transacción
oldbalanceOrg = st.number_input("Saldo previo del emisor (€)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("Saldo posterior del emisor (€)", min_value=0.0, value=9000.0)

# Saldos del receptor antes y después de la transacción
oldbalanceDest = st.number_input("Saldo previo del receptor (€)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("Saldo posterior del receptor (€)", min_value=0.0, value=0.0)

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
if st.button("Predecir"):

    # Construir el DataFrame con las mismas columnas que usó el modelo en entrenamiento
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }])

    # El pipeline aplica automáticamente el preprocesamiento antes de predecir
    prediction = model.predict(input_data)[0]

    st.subheader(f"Resultado de la predicción: `{int(prediction)}`")

    if prediction == 1:
        st.error("⚠️ Esta transacción podría ser fraudulenta.")
    else:
        st.success("✅ Esta transacción parece legítima.")
