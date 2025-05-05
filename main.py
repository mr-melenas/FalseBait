from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from core.config import settings
import scraping as scraping
import asyncio


#inicializamos la app
app = FastAPI(
    title=settings.proyect_name,
    description=settings.description,
    version=settings.version
    )

# rutas static y templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html", 
        {
            "request": request, 
            "title": settings.proyect_name + ", " + settings.version,
            "description": settings.description,
            "api_prefix": settings.api_prefix+settings.api_version,
        }
    )

@app.post(settings.api_prefix+settings.api_version+"/predict")
async def predict(data: dict):
    print(data["url"]) 
    try:
        loop = asyncio.get_event_loop()
        url_clasification = await loop.run_in_executor(
            None,
            scraping.es_url_que_responde, data["url"]
            #scraping.extract_features_from_url, data["url"]
            )
        return {
            "clasificate_url": url_clasification,
            "url": data["url"] # otres datos que quieras devolver
        } 
    except Exception as e: 
        return {
            "error": str(e)
        }   
