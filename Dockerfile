# To change for a new python version if needed
# I am not using a "slim" image because it might cause problems with some of the libraries needed (xgboost definitely, and maybe others)
# The warning I do have in this line is because it's not especified the exact python version like 3.11.9 or such
# Version must be 3.13 otherwise not all packages needed can be installed due to older versions cannot support it
FROM python:3.13


# In order to avoid issues with other operating systems
RUN mkdir -p /tmp/cache/fontconfig && chmod 777 /tmp/cache/fontconfig

# Automatically check for folders
WORKDIR /falsebait

# Copy the requirements file before to take advantage of the cache
COPY requirements.txt .

# Install python
RUN python -m pip install --upgrade pip
# Copy and install the requirements
# --use-pep517 is needed to install some packages that are not compatible with the new version of pip
# --no-cache-dir is used to avoid caching the packages, which can save space
# --timeout 60 is used to avoid timeout issues when installing packages
# -v is used to show the verbose output of the installation process
RUN pip install -r requirements.txt --use-pep517 --no-cache-dir --timeout 60 -v

# Copy the rest of the code
COPY . .

# Expone los puertos:
# - 8000 for FastAPI
# - 8888 forJupyter notebook    JUPYTER causing issues while composing the docker since it seems is not compatible with supabase and supabase is needed for the project
# - 7860 # Gradio
EXPOSE 8000 
# EXPOSE 8888
EXPOSE 7860 

