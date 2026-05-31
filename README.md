# 🔍 Detección de Fraude en Transacciones Financieras

Proyecto de Machine Learning que entrena un modelo de clasificación para detectar transacciones fraudulentas, y lo despliega como una aplicación web interactiva con Streamlit.

---

## 📋 Descripción

Las transacciones fraudulentas representan menos del 0,13% del total de operaciones en el dataset, lo que convierte este problema en un caso típico de **clases desbalanceadas**. El proyecto aborda este desafío mediante regresión logística con pesos de clase balanceados, y expone el modelo a través de una interfaz web sencilla donde el usuario puede introducir los datos de una transacción y obtener una predicción en tiempo real.

---

## 📁 Estructura del proyecto

```
Fraud_detection/
│
├── data/
│   └── AIML Dataset.csv          # Dataset original
│
├── fraud_detection.ipynb         # Notebook: análisis, entrenamiento y exportación del modelo
├── fraud_detection.py            # Aplicación Streamlit
├── fraud_detection_pipeline.pkl  # Pipeline entrenado (generado por el notebook)
└── README.md
```

---

## 📊 Dataset

- **Fuente:** [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) — publicado en Kaggle por Aman Ali Siddiqui
- **Base:** datos sintéticos generados con [PaySim](https://www.researchgate.net/publication/313138956_PAYSIM_A_FINANCIAL_MOBILE_MONEY_SIMULATOR_FOR_FRAUD_DETECTION), un simulador de transacciones de dinero móvil desarrollado con fines de investigación en detección de fraude
- **Volumen:** 6,3 millones de registros
- **Tasa de fraude:** 0,13% (fuerte desbalance de clases)
- **Tipos de transacción:**

| Tipo | Descripción |
|---|---|
| `CASH_IN` | Ingreso de efectivo a través de un comercio (aumenta el saldo) |
| `CASH_OUT` | Retirada de efectivo a través de un comercio (reduce el saldo) |
| `DEBIT` | Envío de dinero desde el servicio móvil a una cuenta bancaria |
| `PAYMENT` | Pago de bienes o servicios a un comercio |
| `TRANSFER` | Transferencia de dinero a otro usuario de la plataforma |

- **Variables utilizadas:**

| Variable | Descripción |
|---|---|
| `type` | Tipo de transacción (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT) |
| `amount` | Importe de la transacción |
| `oldbalanceOrg` | Saldo del emisor antes de la transacción |
| `newbalanceOrig` | Saldo del emisor después de la transacción |
| `oldbalanceDest` | Saldo del receptor antes de la transacción |
| `newbalanceDest` | Saldo del receptor después de la transacción |
| `isFraud` | Variable objetivo (1 = fraude, 0 = legítima) |

> Las columnas `step`, `nameOrig`, `nameDest` e `isFlaggedFraud` fueron descartadas durante el preprocesamiento.

---

## 🧠 Metodología

### 1. Análisis exploratorio (EDA)
- Distribución de tipos de transacción
- Tasa de fraude por tipo de operación (el fraude se concentra en TRANSFER y CASH_OUT)
- Distribución del importe (escala logarítmica)
- Análisis de diferencias de saldo entre emisor y receptor
- Evolución del fraude a lo largo del tiempo (por step)
- Matriz de correlación entre variables numéricas

### 2. Preprocesamiento
- **Variables numéricas:** estandarización con `StandardScaler`
- **Variables categóricas:** codificación con `OneHotEncoder` (drop="first")
- Pipeline con `ColumnTransformer` de scikit-learn

### 3. Modelo
- **Algoritmo:** Regresión Logística
- **Gestión del desbalance:** `class_weight="balanced"`
- **División:** 70% entrenamiento / 30% test con estratificación
- **Exportación:** `joblib` → `fraud_detection_pipeline.pkl`

---

## 🖥️ Aplicación web

La app permite introducir los parámetros de una transacción y obtener una predicción inmediata.

### Ejecutar localmente

```bash
# 1. Instalar dependencias
pip install streamlit pandas scikit-learn joblib

# 2. Lanzar la aplicación
streamlit run fraud_detection.py
```

La app se abrirá automáticamente en `http://localhost:8501`

---

## ⚙️ Requisitos

```
python >= 3.8
streamlit
pandas
scikit-learn
joblib
matplotlib
seaborn
numpy
```

---

## ⚠️ Limitaciones

- El modelo es una regresión logística; modelos más complejos (Random Forest, XGBoost) probablemente mejorarían el rendimiento dado el desbalance de clases.
- El dataset es sintético, por lo que los resultados no son directamente extrapolables a datos reales.
- La app no valida la coherencia entre los saldos introducidos (por ejemplo, que el nuevo saldo sea consistente con el importe de la transacción).

---

## 👤 Autor

Proyecto realizado siguiendo el tutorial de [este vídeo](https://www.youtube.com/watch?v=4Od5_z28iIE&list=PLTsu3dft3CWg69zbIVUQtFSRx_UV80OOg).