#!/bin/bash

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

# Check if requirements are installed
if [ ! -f "requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch requirements_installed
fi

# Run the application
echo "Starting Gradio application..."
python app.py
