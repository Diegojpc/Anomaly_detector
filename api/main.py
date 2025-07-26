# api/main.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

from src.predict import make_prediction
from src.update import incremental_update, NewTransactionData

class Transaction(BaseModel):
    user_id: str
    transaction_type: str
    amount: float
    balance_before: float
    balance_after: float
    timestamp: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Cargando modelos...")
    try:
        app.state.model = joblib.load('models/detector.joblib')
        app.state.online_model = joblib.load('models/online_detector.joblib')
        app.state.scaler = joblib.load('models/scaler.joblib')
        app.state.features = joblib.load('models/features.joblib')
    except FileNotFoundError as e:
        raise RuntimeError(f"Modelos no encontrados. Ejecuta 'python src/model.py'. Error: {e}")
    yield
    print("INFO:     Limpiando recursos...")

app = FastAPI(title="API Híbrida de Detección de Anomalías", version="3.1", lifespan=lifespan)

@app.post("/predict")
async def predict_anomaly(transaction: Transaction, request: Request):
    df = pd.DataFrame([transaction.model_dump()])
    loop = asyncio.get_running_loop()
    
    result = await loop.run_in_executor(
        None, make_prediction, df,
        request.app.state.model, request.app.state.online_model,
        request.app.state.scaler, request.app.state.features
    )
    return result

@app.post("/update")
def update_model(transaction: NewTransactionData, request: Request):
    try:
        message = incremental_update(transaction, request.app.state)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la actualización: {str(e)}")