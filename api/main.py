# api/main.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
from loguru import logger

from src.predict import make_prediction
from src.update import incremental_update, NewTransactionData

# --- Constantes ---
MODELS_DIR = 'models'

# --- Modelos Pydantic ---
class Transaction(BaseModel):
    user_id: str
    transaction_type: str
    amount: float
    balance_before: float
    balance_after: float
    timestamp: datetime

# --- Ciclo de vida de la aplicación (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación. Carga los modelos al iniciar
    y los libera al finalizar.
    """
    logger.info("Cargando modelos...")
    try:
        app.state.model = joblib.load('models/detector.joblib')
        app.state.online_model = joblib.load('models/online_detector.joblib')
        app.state.scaler = joblib.load('models/scaler.joblib')
        app.state.features = joblib.load('models/features.joblib')
        logger.info("Modelos cargados exitosamente.")
    except FileNotFoundError as e:
        raise RuntimeError(f"Modelos no encontrados en '{MODELS_DIR}'. Ejecuta 'python src/model.py'. Error: {e}")
    yield
    logger.info("Limpiando recursos...")

app = FastAPI(
    title="API Híbrida de Detección de Anomalías",
    description="API para detectar transacciones anómalas usando un modelo batch y un modelo online adaptable.",
    version="3.2",
    lifespan=lifespan
)

# --- Endpoints ---
@app.post("/predict")
async def predict_anomaly(transaction: Transaction, request: Request):
    """
    Recibe una transacción y predice si es una anomalía.
    
    Utiliza un enfoque híbrido:
    - Un modelo **batch** (LGBM) robusto.
    - Un modelo **online** (SGD) que se adapta en tiempo real.
    
    La predicción se ejecuta en un hilo separado para no bloquear la API.
    """
    df = pd.DataFrame([transaction.model_dump()])
    loop = asyncio.get_running_loop()
    
    # Ejecuta la función de predicción (que usa CPU) en un pool de hilos
    # para no bloquear el event loop principal de asyncio.
    result = await loop.run_in_executor(
        None,
        make_prediction,
        df,
        request.app.state.model,
        request.app.state.online_model,
        request.app.state.scaler,
        request.app.state.features
    )
    return result

@app.post("/update")
async def update_model(transaction: NewTransactionData, request: Request):
    """
    Actualiza el modelo online con una nueva transacción etiquetada.
    
    Esta operación es rápida y permite que el modelo se adapte a nuevos
    patrones sin necesidad de un re-entrenamiento completo.
    """
    loop = asyncio.get_running_loop()
    try:
        # Ejecuta la función de actualización en un pool de hilos separado
        # para no bloquear el event loop principal de asyncio.
        message = await loop.run_in_executor(
            None,
            incremental_update,
            transaction,
            request.app.state
        )
        return {"message": message}
    except Exception as e:
        # Loggear el error sería ideal en producción
        logger.error(f"Falló la actualización incremental: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la actualización: {str(e)}")