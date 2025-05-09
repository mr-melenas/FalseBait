import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from main import app

@pytest.mark.asyncio
async def test_end_to_end():
    """
    Test completo del flujo:
    1. Cargar el endpoint /api/v1/predict de FastApi
    2. Ingresa una url en el input para clasificar
    3. Verifica código 200
    4. Verifica estructura de respuesta y que devuelve una respuesta de clasificación
    """

    # Evita levantar un servidor real en pruebas, pero simula todas las rutas como si lo hiciera
    transport = ASGITransport(app=app)

    # Simula el cliente HTTP, base_url es una url simbólica
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/v1/predict", json={"url": "https://google.com"})

    assert response.status_code == 200

    data = response.json()
    assert "url" in data # verificamos que la data de respuesta contenga la url
    assert "classification" in data # verificamos que la data de respuesta contenga la clasificacion
    assert data["url"] == "https://google.com" # verificamos que la url de respuesta sea la misma que la que enviamos
    assert data["classification"] in ["Legítimo", "Phishing", "Desconocido"] # verificamos que la clasificacion sea una de las tres posibles
