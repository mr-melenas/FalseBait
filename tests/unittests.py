import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import settings
from fastapi.testclient import TestClient
from scraping import es_url_que_responde

from main import app

class UnitTestExample(unittest.TestCase):
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


# comprobar que el servidor de FastApi está corriendo
class TestFastAPIServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_legitimate_url(self):
        response = self.client.post(settings.api_prefix + settings.api_version + "/predict", json={"url": "https://www.google.com"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("classification", data)
        self.assertEqual(data["url"], "https://www.google.com")

if __name__ == "__main__":
    unittest.main()