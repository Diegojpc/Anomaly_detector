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
from loguru import logger


warnings.filterwarnings('ignore')

# --- Constantes de configuración ---
DATA_PATH = 'data/data_transactions.csv'
MODEL_DIR = 'models/'
EVAL_REPORT_PATH = os.path.join(MODEL_DIR, 'evaluation_report.txt')

# --- Lógica de Ingeniería de Características ---
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la ingeniería de características sobre el DataFrame de transacciones.

    Args:
        df (pd.DataFrame): DataFrame con los datos de las transacciones.

    Returns:
        pd.DataFrame: DataFrame con las nuevas características añadidas.
    """
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

    # Características temporales
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
    df_copy['day_of_month'] = df_copy['timestamp'].dt.day

    # Característica de tipo de transacción
    df_copy['type_recarga'] = (df_copy['transaction_type'] == 'recarga').astype(int)

    # Características de consistencia del balance
    recarga_mask = df_copy['type_recarga'] == 1
    retiro_mask = df_copy['type_recarga'] == 0
    
    expected_balance = df_copy['balance_before'].copy()
    expected_balance[recarga_mask] += df_copy['amount'][recarga_mask]
    expected_balance[retiro_mask] -= df_copy['amount'][retiro_mask]

    df_copy['balance_diff_error'] = abs(df_copy['balance_after'] - expected_balance)

    # Característica de sobregiro
    df_copy['is_overdraft'] = (df_copy['balance_after'] < 0).astype(int)
    
    return df_copy

# --- Lógica de Entrenamiento ---
def train():
    """
    Ejecuta el pipeline completo de entrenamiento: carga de datos, ingeniería de
    características, entrenamiento de modelos (batch y online), evaluación y
    guardado de artefactos y reporte.
    """
    logger.info("Iniciando pipeline de entrenamiento...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Carga y preparación de datos    
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

    # 2. Escalado y sobremuestreo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # 3. Entrenar y evaluar modelo Batch (LGBM)
    logger.info("Entrenando modelo Batch (LightGBM)...")
    batch_model = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42)
    batch_model.fit(X_train_resampled, y_train_resampled)

    y_pred_batch = (batch_model.predict_proba(X_test_scaled)[:, 1] > 0.5).astype(int)
    auc_batch = roc_auc_score(y_test, batch_model.predict_proba(X_test_scaled)[:, 1])
    report_batch = classification_report(y_test, y_pred_batch)

    # Entrenar modelo Online inicial (SGD)
    online_model = SGDClassifier(loss='log_loss', random_state=42)
    online_model.fit(X_train_resampled, y_train_resampled)

    y_pred_online = (online_model.predict_proba(X_test_scaled)[:, 1] > 0.5).astype(int)
    auc_online = roc_auc_score(y_test, online_model.predict_proba(X_test_scaled)[:, 1])
    report_online = classification_report(y_test, y_pred_online)
    
    
    # 5. Guardar artefactos (modelos, scaler, features)
    logger.info(f"Guardando artefactos en '{MODEL_DIR}'...")
    joblib.dump(batch_model, os.path.join(MODEL_DIR, 'detector.joblib'))
    joblib.dump(online_model, os.path.join(MODEL_DIR, 'online_detector.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    joblib.dump(features, os.path.join(MODEL_DIR, 'features.joblib'))
    
    # 6. Guardar reporte de evaluación
    logger.info(f"Guardando reporte de evaluación en '{EVAL_REPORT_PATH}'...")
    with open(EVAL_REPORT_PATH, 'w') as f:
        f.write("="*60 + "\n")
        f.write("        REPORTE DE EVALUACIÓN DE MODELOS\n")
        f.write("="*60 + "\n\n")
        
        f.write("--- MODELO BATCH (LightGBM) ---\n")
        f.write(f"AUC-ROC Score: {auc_batch:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report_batch)
        f.write("\n" + "-"*60 + "\n\n")

        f.write("--- MODELO ONLINE (SGDClassifier) ---\n")
        f.write(f"AUC-ROC Score: {auc_online:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report_online)
        f.write("\n" + "="*60 + "\n")

    logger.info("Pipeline de entrenamiento completado exitosamente.")
    logger.info(f"Reporte de evaluación disponible en '{EVAL_REPORT_PATH}'.")


if __name__ == "__main__":
    train()