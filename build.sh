#!/bin/bash
# build.sh — local build + run script
set -e
echo "Installing Python deps..."
pip install -r requirements.txt
echo "Building React frontend..."
cd frontend && npm install && npm run build && cd ..
echo "Starting NeuroScan AI on http://localhost:8000"
python run.py
