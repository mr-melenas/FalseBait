# To change for a new python version if needed
# I am not using a "slim" image because it might cause problems with some of the libraries needed (xgboost definitely, and maybe others)
FROM python:3.11 

# Automatically check for folders
WORKDIR /app 

# Copy and install the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# # Copy the host and app folders to the container 
COPY . .


# Expone los puertos:
# - 8000 for FastAPI
# - 8888 forJupyter notebook 
# - 7860 # Gradio
EXPOSE 8000 
EXPOSE 8888
EXPOSE 7860 

# To start fastapi
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
