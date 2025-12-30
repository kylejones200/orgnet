#!/bin/bash
# Start the Flask web application

echo "Starting Email Analytics Platform..."
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run setup_venv.sh first."
    exit 1
fi

# Check if model exists
if [ ! -d "model_output" ] || [ -z "$(ls -A model_output)" ]; then
    echo "Warning: No model found in model_output/"
    echo "Please build a model first using the pipeline."
    exit 1
fi

# Start Flask app
echo "Starting web application on http://localhost:5001"
echo "Press CTRL+C to stop"
echo ""
python3 app.py model_output 5001

