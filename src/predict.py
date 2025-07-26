# src/predict.py
import pandas as pd
from .model import create_features

def make_prediction(
        input_data: pd.DataFrame,
        batch_model, online_model,
        scaler, features
    ) -> dict:
    """
    Realiza una predicción de anomalía utilizando un enfoque híbrido.

    Combina las predicciones de un modelo batch (estático) y un modelo
    online (adaptable). Una transacción se considera anómala si cualquiera
    de los dos modelos la marca como tal.

    Args:
        input_data (pd.DataFrame): DataFrame con la transacción a evaluar.
        batch_model: Modelo de machine learning pre-entrenado (ej. LGBM).
        online_model: Modelo de machine learning adaptable (ej. SGDClassifier).
        scaler: Objeto StandardScaler ajustado.
        features (list): Lista de nombres de las características a usar.

    Returns:
        dict: Un diccionario con el resultado de la predicción, incluyendo 
        el score de anomalía y el detalle de cada modelo.
    """
    # 1. Aplicar la misma ingeniería de características que en el entrenamiento
    df_featured = create_features(input_data)
    X_input = df_featured[features]

    # 2. Escalar los datos
    X_scaled = scaler.transform(X_input)
    
    # 3. Obtener predicciones de ambos modelos
    pred_batch = batch_model.predict(X_scaled)[0]
    prob_batch = batch_model.predict_proba(X_scaled)[0][1]
    
    pred_online = online_model.predict(X_scaled)[0]
    prob_online = online_model.predict_proba(X_scaled)[0][1]

    # 4. Combinar resultados (estrategia conservadora: si uno dice que es anomalía, lo es)
    is_anomaly = bool(pred_batch or pred_online)
    anomaly_score = max(float(prob_batch), float(prob_online))
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 4),
        "details": {
            "batch_model_prediction": bool(pred_batch),
            "batch_model_score": round(float(prob_batch), 4),
            "online_model_prediction": bool(pred_online),
            "online_model_score": round(float(prob_online), 4),
        }
    }