# src/update.py
import os
import pandas as pd
import joblib
from pydantic import BaseModel
from datetime import datetime
from .predict import create_features # Reutilizamos la función de predict.py

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
    Actualiza el modelo online (SGDClassifier) con una nueva transacción etiquetada.
    Esta operación es muy rápida y se realiza en tiempo real.
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

        # --- CORRECCIÓN APLICADA AQUÍ ---
        # Al llamar a partial_fit, siempre debemos especificar todas las clases posibles
        # del problema de clasificación para que el modelo se inicialice correctamente.
        online_model.partial_fit(X_new_scaled, y_new.values, classes=[0, 1])
        # ------------------------------------

        # Guardar el modelo online actualizado para persistencia
        joblib.dump(online_model, os.path.join('models', 'online_detector.joblib'))
        
        return "Modelo online actualizado exitosamente con la nueva transacción."

    except Exception as e:
        print(f"ERROR: Falló la actualización incremental: {e}")
        raise