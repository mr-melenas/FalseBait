# To change for a new python version if needed
# I am not using a "slim" image because it might cause problems with some of the libraries needed (xgboost definitely, and maybe others)
# The warning I do have in this line is because it's not especified the exact python version like 3.11.9 or such
# Version must be 3.13 otherwise not all packages needed can be installed due to older versions cannot support it
FROM python:3.13


# In order to avoid issues with other operating systems
RUN mkdir -p /tmp/cache/fontconfig && chmod 777 /tmp/cache/fontconfig

# Automatically check for folders
WORKDIR /falsebait

# # Copy the host and app folders and files into the container 
COPY . .

# Install python
RUN python -m pip install --upgrade pip

# Copy and install the requirements
RUN pip install --no-cache-dir -r requirements.txt


# Expone los puertos:
# - 8000 for FastAPI
# - 8888 forJupyter notebook 
# - 7860 # Gradio
EXPOSE 8000 
EXPOSE 8888
EXPOSE 7860 


# # Healthcheck to ensure the server is running properly, it checks the endpoints 3 times every 30 seconds and if it is ok then is healthy and otherwise not
# HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
#   CMD curl --fail http://0.0.0.0:8000/docs || exit 1
