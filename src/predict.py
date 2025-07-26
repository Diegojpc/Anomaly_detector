# src/predict.py
import pandas as pd
import numpy as np

# Duplicamos la función de creación de características aquí
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy['hour'], df_copy['day_of_week'], df_copy['day_of_month'] = df_copy['timestamp'].dt.hour, df_copy['timestamp'].dt.dayofweek, df_copy['timestamp'].dt.day
    df_copy['type_recarga'] = (df_copy['transaction_type'] == 'recarga').astype(int)
    recarga_mask = df_copy['type_recarga'] == 1
    retiro_mask = df_copy['type_recarga'] == 0
    expected_balance = df_copy['balance_before'].copy()
    expected_balance[recarga_mask] += df_copy['amount'][recarga_mask]
    expected_balance[retiro_mask] -= df_copy['amount'][retiro_mask]
    df_copy['balance_diff_error'] = np.abs(df_copy['balance_after'] - expected_balance)
    df_copy['is_overdraft'] = (df_copy['balance_after'] < 0).astype(int)
    return df_copy

def make_prediction(input_data: pd.DataFrame, batch_model, online_model, scaler, features) -> dict:
    df_featured = create_features(input_data)
    X_input = df_featured[features]
    X_scaled = scaler.transform(X_input)
    
    pred_batch = batch_model.predict(X_scaled)[0]
    prob_batch = batch_model.predict_proba(X_scaled)[0][1]
    
    pred_online = online_model.predict(X_scaled)[0]
    prob_online = online_model.predict_proba(X_scaled)[0][1]

    is_anomaly = bool(pred_batch or pred_online)
    anomaly_score = max(float(prob_batch), float(prob_online))
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "details": {
            "batch_model_prediction": bool(pred_batch),
            "online_model_prediction": bool(pred_online)
        }
    }