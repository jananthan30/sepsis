#!/bin/bash

# Start FastAPI in background on port 8000
echo "Starting FastAPI server on port 8000..."
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to start
sleep 3

# Start Streamlit on port 7860 (HuggingFace Spaces default)
echo "Starting Streamlit app on port 7860..."
python -m streamlit run streamlit_app.py --server.port=7860 --server.address=0.0.0.0
