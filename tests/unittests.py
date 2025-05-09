import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import settings
from fastapi.testclient import TestClient
from scraping import es_url_que_responde

from main import app
import joblib

class UnitTestExample(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_legitimate_url(self):
        # comporobamos que al pasarle una url valida nos devuleva True porque es legitima
        url = "https://www.google.com"
        resultado = es_url_que_responde(url)
        self.assertEqual(resultado, True)

    def test_phishing_url(self):
        # comporobamos que al pasarle una url de phishing nos devuleva False porque es phishing
        url = "http://0123656.com/mobile-client/index/index.html"
        resultado = es_url_que_responde(url)
        self.assertEqual(resultado, False)

    def test_not_existent_url(self):
        # comporobamos que al pasarle una url inválida/que no existe nos devuleva False
        url = "https://www.google.tr"
        resultado = es_url_que_responde(url)
        self.assertEqual(resultado, False)

    def test_empty_url(self):
        # comprobamos que al pasarle una url vacia nos devuleva False
        url = ""
        resultado = es_url_que_responde(url)
        self.assertEqual(resultado, False)

    def test_FastApi_is_working(self):
        # comprobar que el servidor de FastApi está corriendo
        response = self.client.post(settings.api_prefix + settings.api_version + "/predict", json={"url": "https://www.google.com"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("classification", data)
        self.assertEqual(data["url"], "https://www.google.com")
    
    def test_model_is_loading(self):
        # comprobamos que el modelo se carga correctamente
        model_path = settings.model_path
        self.assertTrue(os.path.exists(model_path), "El modelo no se encuentra en la ruta especificada: {model_path}")

        try:
            model = joblib.load(model_path)
            self.assertIsNotNone(model, "El modelo no se ha cargado correctamente.")
        except Exception as e:
            self.fail(f"Error al cargar el modelo: {e}")

if __name__ == "__main__":
    unittest.main()