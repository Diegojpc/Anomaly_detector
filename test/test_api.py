# test/test_api.py
import pytest
from fastapi.testclient import TestClient # <--- CAMBIO 1: Importar TestClient
from datetime import datetime
import os
import sys

# Añadir el directorio raíz al path para que el test pueda encontrar la app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

# CAMBIO 2: Usar una fixture síncrona
@pytest.fixture(scope="module")
def client():
    """Crea un cliente de prueba síncrono para la API."""
    # El TestClient sí acepta el argumento 'app'
    with TestClient(app) as c:
        yield c

# CAMBIO 3: Convertir todas las pruebas a funciones síncronas (quitar async y @pytest.mark.asyncio)
def test_predict_normal_transaction(client: TestClient):
    """Prueba que una transacción claramente normal se clasifique como no anómala."""
    normal_transaction = {
        "user_id": "user_1",
        "transaction_type": "recarga",
        "amount": 150.00,
        "balance_before": 1000.00,
        "balance_after": 1150.00,
        "timestamp": datetime.now().isoformat()
    }
    # CAMBIO 4: Quitar 'await' de la llamada a la API
    response = client.post("/predict", json=normal_transaction)
    
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert data["is_anomaly"] is False
    assert "anomaly_score" in data
    assert data["anomaly_score"] < 0.5

def test_predict_anomalous_transaction(client: TestClient):
    """Prueba que una transacción claramente anómala se clasifique como tal."""
    anomalous_transaction = {
        "user_id": "user_fraud",
        "transaction_type": "retiro",
        "amount": 5000000.00,
        "balance_before": 1000.00,
        "balance_after": -4999000.00,
        "timestamp": datetime.now().isoformat()
    }
    response = client.post("/predict", json=anomalous_transaction)
    
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is True
    assert data["anomaly_score"] > 0.5

def test_predict_invalid_data(client: TestClient):
    """Prueba que la API maneje correctamente datos de entrada inválidos."""
    invalid_transaction = {
        "user_id": "user_1",
        "transaction_type": "recarga",
        "amount": "not_a_float",
        "balance_before": 1000.00,
        "balance_after": 1150.00,
        "timestamp": "not_a_date"
    }
    response = client.post("/predict", json=invalid_transaction)
    
    assert response.status_code == 422

def test_update_endpoint(client: TestClient):
    """Prueba el endpoint /update para asegurar que el modelo online se actualiza."""
    new_transaction = {
        "user_id": "user_new",
        "transaction_type": "recarga",
        "amount": 250.00,
        "balance_before": 500.00,
        "balance_after": 750.00,
        "timestamp": datetime.now().isoformat(),
        "is_anomaly": 0
    }

    # Asegurarse de que el modelo online_detector.joblib existe
    if not os.path.exists('models/online_detector.joblib'):
        pytest.fail("El modelo 'online_detector.joblib' no existe. Asegúrate de haber entrenado el modelo.")
        
    response = client.post("/update", json=new_transaction)
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Modelo online actualizado" in data["message"]