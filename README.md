# 🔍 FalseBait: Detección de Phishing con Machine Learning

## 📌 Índice

- [Acerca del Proyecto](#acerca-del-proyecto)
- [Características Principales](#características-principales)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Dependencias](#dependencias)
- [Instalación y Uso](#instalación-y-uso)
- [API Endpoints](#api-endpoints)
- [Flujo de Clasificación de URLs](#flujo-de-clasificación-de-urls)
- [Posibles Mejoras](#posibles-mejoras)

## Acerca del Proyecto

FalseBait es una plataforma web que analiza y clasifica textos en función de su veracidad. El proyecto utiliza una arquitectura de microservicios con contenedores Docker, persistencia en Supabase, y despliegue en la nube mediante Render.

Cuenta con una funcionalidad clave: reentrenamiento automático del modelo con nuevas URLs insertadas por los usuarios, lo que permite una mejora progresiva del sistema. Además, incluye logs de actividad para trazabilidad y depuración.

Accede a la app online:
🔗 https://falsebaitgradio.onrender.com/

Accede a documentación del proyecto: 
🔗 https://deepwiki.com/mr-melenas/FalseBait/1-overview

## Características Principales

✅ **Interfaz Dual**: Acceso tanto a través de interfaz web (Gradio) como API (FastAPI)

✅ **Clasificación basada en ML**: Utiliza dos modelos para redundancia y precisión

✅ **Extracción de Características**: Analiza estructura de URL, propiedades de dominio y contenido

✅ **Almacenamiento de Resultados**: Guarda resultados de clasificación en base de datos Supabase

✅ **Análisis Profundo**: Extrae más de 40 características de cada URL para una clasificación precisa

✅ **Despliegue Containerizado**: Fácil despliegue mediante Docker y Docker Compose

## Arquitectura del Sistema

```
POST /api/v1/predict → Web requests → Model prediction → extract_features_from_url() → save_fill_complete()

Usuario → Gradio UI (gradio_ui.py) → API Client → FastAPI Backend (main.py) → URL Predictor (scraping.py) → Web Scraper (scraping.py) → ML Models (model_clf_A.pkl, model_clf_B.pkl) → Supabase Database (supabase_db.py)
```

FalseBait sigue una arquitectura de microservicios con clara separación entre UI, lógica de aplicación y almacenamiento de datos. El sistema consta de:

- **Gradio UI**: Interfaz web para enviar URLs para clasificación
- **FastAPI Backend**: Procesa solicitudes de clasificación de URL a través de una API REST
- **URL Predictor**: Componente principal que maneja el análisis de URL
- **Feature Extractor**: Extrae características relevantes de las URLs para análisis
- **ML Models**: Dos modelos de clasificación que determinan si una URL es legítima o de phishing
- **Supabase Database**: Almacena resultados de clasificación y características extraídas


## Tecnologías Utilizadas

![Render](https://img.shields.io/badge/-Render-46B7C8?logo=render&logoColor=white) 
![Gradio](https://img.shields.io/badge/-Gradio-FFB400?logo=python&logoColor=black)  
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) 
![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white)   
![Jupyter](https://img.shields.io/badge/-Jupyter-FF3C00?logo=jupyter&logoColor=white)  
![Supabase](https://img.shields.io/badge/-Supabase-3ECF8E?logo=supabase&logoColor=white)  
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white)

## Dependencias

![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)  
![Logging](https://img.shields.io/badge/-Logging-4B8BBE?logo=python&logoColor=white)  
![pytest](https://img.shields.io/badge/-pytest-0A9EDC?logo=pytest&logoColor=white)  
![Uvicorn](https://img.shields.io/badge/-Uvicorn-7A2A8B?logo=uvicorn&logoColor=white)  
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)  
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)  
![BeautifulSoup](https://img.shields.io/badge/-BeautifulSoup-8B2A2A?logo=python&logoColor=white)


## Instalación y Uso

### 1️⃣ Clonar el Repositorio
```bash
git clone https://github.com/mr-melenas/FalseBait
cd FalseBait
```

### 2️⃣ Configuración con Docker Compose
```bash
    docker compose up --build
```

Esto iniciará tres servicios:
- Servicio Gradio UI: Expone la interfaz web en el puerto 7860
- Servicio FastAPI: Proporciona la API backend en el puerto 8000
- Servicio Test Runner: Ejecuta pruebas para verificar la funcionalidad del sistema

### 3️⃣ Acceder a la Aplicación
🌐 Opción 1: Usar la versión online
    Accede directamente a la aplicación en la nube:
    https://falsebaitgradio.onrender.com/

🧪 Opción 2: Ejecutar en local con Docker
Asegúrate de tener Docker instalado.
1️⃣ Descargar las imágenes necesarias:
```
    docker pull yaelpc/falsebait-fastapi:v7  
    docker pull yaelpc/falsebait-gradio:v7  
    docker pull yaelpc/falsebait-tests:v7
```

2️⃣ Ejecutar los contenedores (ejemplo básico):
```
    docker run -d -p 8000:8000 yaelpc/falsebait-fastapi:v7  
    docker run -d -p 7860:7860 yaelpc/falsebait-gradio:v7 
```

3️⃣ Acceder a la app localmente:
http://127.0.0.1:7860
- **API**: Disponible en `http://localhost:8000/api/v1/predict`

## API Endpoints
    https://falsebait-ake4.onrender.com/docs

### POST /api/v1/predict

Este endpoint acepta una URL y devuelve su clasificación como legítima, phishing o desconocida.

**Ejemplo de solicitud:**
```json
{
  "url": "https://example.com"
}
```

**Ejemplo de respuesta:**
```json
{
  "url": "https://example.com",
  "classification": "Legítimo"
}
```

## Flujo de Clasificación de URLs

El proceso de clasificación de URL sigue estos pasos:

1. Un usuario envía una URL a través de la UI de Gradio o directamente a la API
2. El backend FastAPI recibe la solicitud y la procesa
3. El sistema verifica si la URL responde
4. Si responde, se extraen características de la URL
5. Se selecciona aleatoriamente un modelo ML (A o B) para clasificar la URL
6. Los resultados de clasificación se almacenan en la base de datos Supabase
7. El resultado ("Legítimo", "Phishing" o "Desconocido") se devuelve al usuario

## Posibles Mejoras

✅ Integrar datasets adicionales para obtener insights más profundos

✅ Desarrollar un pipeline automatizado para actualizaciones de datos

✅ Mejorar visualizaciones con dashboards interactivos

✅ Implementar modelos de machine learning para predicción de tendencias

✅ Añadir más características de análisis para mejorar la precisión

---

🧑‍💻 ## Colaboradores

Este proyecto ha sido desarrollado por:

- Yael Parra  [Linkedin](https://www.linkedin.com/in/yael-parra/) [Github](https://github.com/Yael-Parra)
- Juan Carlos Macías [Linkedin](https://www.linkedin.com/in/juancarlosmacias/) [Github](https://github.com/juancmacias)
- Alla Haruty [Linkedin](https://www.linkedin.com/in/allaharuty/) [Github](https://github.com/alharuty)
- Max Beltran [Linkedin](https://www.linkedin.com/in/max-beltran/) [Github](https://github.com/mr-melenas)


Desarrollado con ❤️ por el grupo II para FactoriaF5.