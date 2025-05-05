# To change for a new python version if needed
# I am not using a "slim" image because it might cause problems with some of the libraries needed (xgboost definitely, and maybe others)
FROM python:3.11


# Copy and install the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the app and model folders to the container THIS WILL CHANGE BECAUSE THE DOCKER IS BEING DONE BEFORE THE APP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
COPY ./app ./app
COPY ./model ./model

# Expone dos puertos:
# - 8000 for FastAPI
# - 8888 forJupyter notebook 
EXPOSE 8000
EXPOSE 8888

# To start fastapi
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
