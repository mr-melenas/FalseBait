#!/bin/bash

# start FastAPI in second plane
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Gradio UI in first plane
python gradio_ui.py 
