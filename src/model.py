# src/model.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# --- Lógica de Ingeniería de Características ---
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
    df_copy['day_of_month'] = df_copy['timestamp'].dt.day

    df_copy['type_recarga'] = (df_copy['transaction_type'] == 'recarga').astype(int)

    recarga_mask = df_copy['type_recarga'] == 1
    retiro_mask = df_copy['type_recarga'] == 0
    
    expected_balance = df_copy['balance_before'].copy()
    expected_balance[recarga_mask] += df_copy['amount'][recarga_mask]
    expected_balance[retiro_mask] -= df_copy['amount'][retiro_mask]

    df_copy['balance_diff_error'] = abs(df_copy['balance_after'] - expected_balance)
    df_copy['is_overdraft'] = (df_copy['balance_after'] < 0).astype(int)
    
    return df_copy

# --- Lógica de Entrenamiento ---
def train():
    print("Iniciando pipeline de entrenamiento...")
    
    DATA_PATH = 'data/data_transactions.csv'
    MODEL_DIR = 'models/'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    data = pd.read_csv(DATA_PATH)
    data_featured = create_features(data)

    features = [
        'amount', 'balance_before', 'balance_after', 'hour', 
        'day_of_week', 'day_of_month', 'type_recarga', 
        'balance_diff_error', 'is_overdraft'
    ]
    target = 'is_anomaly'

    X = data_featured[features]
    y = data_featured[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Entrenar modelo Batch (LGBM)
    model = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Entrenar modelo Online inicial (SGD)
    online_model = SGDClassifier(loss='log_loss', random_state=42)
    online_model.fit(X_train_resampled, y_train_resampled)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    print("\n--- Reporte de Evaluación (Modelo Batch LGBM) ---")
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))

    # Guardar artefactos
    joblib.dump(model, f'{MODEL_DIR}detector.joblib')
    joblib.dump(online_model, f'{MODEL_DIR}online_detector.joblib')
    joblib.dump(scaler, f'{MODEL_DIR}scaler.joblib')
    joblib.dump(features, f'{MODEL_DIR}features.joblib')
    
    print(f"\nModelos (batch y online) guardados en '{MODEL_DIR}'.")

if __name__ == "__main__":
    train()