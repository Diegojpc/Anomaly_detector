# Sistema de Detección de Anomalías en Transacciones

Este proyecto implementa una solución completa de Machine Learning para detectar transacciones financieras anómalas en tiempo real. Utiliza una arquitectura híbrida que combina un modelo robusto entrenado por lotes (Batch) con un modelo de aprendizaje incremental (Online) para una rápida adaptación a nuevos patrones.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/tested%20with-pytest-red.svg)](https://docs.pytest.org/en/latest/)

---

## Características Principales

-   **Modelo Híbrido:** Combina un modelo **LightGBM** (Batch) para una alta precisión general con un **SGDClassifier** (Online) que aprende de cada nueva transacción.
-   **Aprendizaje Incremental Real:** El endpoint `/update` actualiza el modelo online en tiempo real con el método `partial_fit`, sin necesidad de reentrenar todo el sistema.
-   **API de Alto Rendimiento:** Construida con **FastAPI**, utilizando `lifespan` para una gestión eficiente de los recursos y `async` para manejar concurrencia.
-   **Manejo de Datos Desbalanceados:** Se utiliza la técnica **SMOTE** (Synthetic Minority Over-sampling Technique) durante el entrenamiento para asegurar que el modelo aprenda adecuadamente de la clase minoritaria (anomalías) y los datos sean mas balanceado a la hora de entrenar.
-   **Ingeniería de Características:** Crea nuevas variables predictivas a partir de los datos crudos, como errores de consistencia en el balance y detección de sobregiros.
-   **Código Modular y Limpio:** Sigue las mejores prácticas de desarrollo de software, con una estructura de proyecto clara y lógica de negocio separada de la API.
-   **Probado:** Incluye una suite de pruebas unitarias con `pytest` para garantizar la fiabilidad de los endpoints.

---

## Estructura del Proyecto

El proyecto sigue una estructura modular para facilitar su mantenimiento y escalabilidad:

```
anomaly_detector/
├── api/
│ └── main.py # Lógica de la API (endpoints, lifespan)
├── data/
│ └── data_transactions.csv # Conjunto de datos de entrenamiento inicial
├── models/
│ └── (vacío inicialmente) # Directorio donde se guardan los modelos serializados
├── src/
│ ├── model.py # Pipeline de entrenamiento (incluye SMOTE)
│ ├── predict.py # Lógica de inferencia del modelo híbrido
│ └── update.py # Lógica de actualización incremental del modelo online
├── test/
│ └── test_api.py # Pruebas unitarias para los endpoints de la API
├── README.md # Este archivo
└── requirements.txt # Dependencias del proyecto
```

---

## Guía de Instalación y Uso

### 1. Prerrequisitos

-   Python 3.9 o superior.
-   `pip` y `venv` para la gestión de paquetes y entornos virtuales.

### 2. Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/Diegojpc/Anomaly_detector.git
    cd Anomaly_detector
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Entrenamiento del Modelo

Antes de poder ejecutar la API, es necesario entrenar los modelos iniciales. Este proceso leerá los datos de `data/data_transactions.csv`, aplicará la ingeniería de características, balanceará los datos con SMOTE y guardará los artefactos del modelo (modelos, escalador, etc.) en la carpeta `models/`.

```bash
python src/model.py
```

### 4. Ejecución de la API

Una vez que los modelos han sido entrenados, puedes iniciar el servidor de la API:

```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- --host 0.0.0.0: Hace que la API sea accesible desde fuera de localhost.
- --port 8000: Especifica el puerto.
- --reload: El servidor se reiniciará automáticamente si detecta cambios en el código.

La API estará disponible en http://localhost:8000.

### 5. Ejecución de Pruebas Unitarias

Para verificar que todos los componentes de la API funcionan correctamente, ejecuta la suite de pruebas unitarias. Asegúrate de haber entrenado los modelos primero.

```
pytest
```

## Uso de la API

Puedes interactuar con la API a través de la documentación interactiva (Swagger UI) que se genera automáticamente o usando herramientas como curl o Postman.

**Documentación Interactiva:** Abre tu navegador y ve a http://localhost:8000/docs.

### Endpoint POST /predict

Este endpoint recibe una transacción y devuelve si es una anomalía, junto con un score de confianza.

Ejemplo de Petición (curl):

```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "user_test",
  "transaction_type": "recarga",
  "amount": 123.45,
  "balance_before": 1000,
  "balance_after": 1123.45,
  "timestamp": "2024-05-21T15:30:00"
}'
```

Ejemplo de Respuesta:

```
{
  "is_anomaly": false,
  "anomaly_score": 0.015,
  "details": {
    "batch_model_prediction": false,
    "online_model_prediction": false
  }
}
```

##  Endpoint POST /update

Este endpoint recibe una transacción ya etiquetada y la utiliza para actualizar el modelo online de forma incremental y en tiempo real.

Ejemplo de Petición (curl):

```
curl -X 'POST' \
  'http://localhost:8000/update' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "user_feedback",
  "transaction_type": "retiro",
  "amount": 5000,
  "balance_before": 100,
  "balance_after": -4900,
  "timestamp": "2024-05-21T16:00:00",
  "is_anomaly": 1
}'
```

Ejemplo de Respuesta:

```
{
  "message": "Modelo online actualizado exitosamente con la nueva transacción."
}
```