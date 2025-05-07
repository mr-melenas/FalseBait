# To change for a new python version if needed
# I am not using a "slim" image because it might cause problems with some of the libraries needed (xgboost definitely, and maybe others)
# The warning I do have in this line is because it's not especified the exact python version like 3.11.9 or such
# Version must be 3.13 otherwise not all packages needed can be installed due to older versions cannot support it
FROM python:3.13

# Automatically check for folders
WORKDIR /falsebait

# # Copy the host and app folders to the container 
COPY . .

# Copy and install the requirements
RUN pip install --no-cache-dir -r requirements.txt


# Expone los puertos:
# - 8000 for FastAPI
# - 8888 forJupyter notebook 
# - 7860 # Gradio
EXPOSE 8000 
EXPOSE 8888
EXPOSE 7860 

# To start fastapi
CMD ["./start.sh"]
