from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import scraping
from core.config import settings
import model_b as modelB
import nest_asyncio
from contextlib import asynccontextmanager

# Carga asincrona para evitar el error de bucle de eventos
nest_asyncio.apply()

class PredictRequest(BaseModel):
    url: str

# Define el ciclo de vida de la app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar la app
    asyncio.create_task(modelB.load_data())
    yield


app = FastAPI(
    title=settings.proyect_name,
    description=settings.description,
    version=settings.version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post(settings.api_prefix + settings.api_version + "/predict")
async def predict(data: PredictRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, scraping.es_url_que_responde, data.url)

        if result is False:
            return {
                "url": data.url,
                "classification": "Desconocido",
                "error": "URL no respondió o falló la extracción"
            }

        classification = "Legítimo" if result == 1 else "Phishing"
        return {
            "url": data.url,
            "classification": classification
        }

    except Exception as e:
        return {
            "url": data.url,
            "classification": "Unknown",
            "error": str(e)
        }
    

