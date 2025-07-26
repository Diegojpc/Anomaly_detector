# src/update.py
import os
import pandas as pd
import joblib
from pydantic import BaseModel
from datetime import datetime
from .model import create_features
from loguru import logger

# Usar la misma constante que en model.py para consistencia
MODELS_DIR = 'models'
ONLINE_MODEL_PATH = os.path.join(MODELS_DIR, 'online_detector.joblib')

class NewTransactionData(BaseModel):
    user_id: str
    transaction_type: str
    amount: float
    balance_before: float
    balance_after: float
    timestamp: datetime
    is_anomaly: int

def incremental_update(transaction: NewTransactionData, app_state):
    """
    Actualiza el modelo online (SGDClassifier) con una nueva transacción
    etiquetada y lo guarda en disco.

    Args:
        transaction (NewTransactionData): Los datos de la nueva transacción
        incluyendo su etiqueta de anomalía.
        app_state: El estado de la aplicación FastAPI (`request.app.state`)
        que contiene los modelos y el scaler cargados.

    Returns:
        str: Un mensaje de confirmación.
    
    Raises:
        Exception: Si ocurre algún error durante el proceso de actualización.
    """
    try:
        # Cargar los componentes necesarios desde el estado de la aplicación
        online_model = app_state.online_model
        scaler = app_state.scaler
        features = app_state.features

        # Preparar los datos de la nueva transacción
        df_new = pd.DataFrame([transaction.model_dump()])
        df_featured = create_features(df_new)
        
        X_new = df_featured[features]
        y_new = df_featured['is_anomaly']

        # Escalar la nueva muestra
        X_new_scaled = scaler.transform(X_new)

        # Actualizar el modelo online con la nueva transacción
        online_model.partial_fit(X_new_scaled, y_new.values, classes=[0, 1])

        # Guardar el modelo online actualizado para persistencia
        joblib.dump(online_model, os.path.join('models', 'online_detector.joblib'))
        
        return "Modelo online actualizado exitosamente con la nueva transacción."

    except Exception as e:
        logger.error(f"Falló la actualización incremental: {e}")
        raise