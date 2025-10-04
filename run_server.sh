#!/bin/bash
# Run script for Secure Heart Attack Prediction System
# This ensures the system starts correctly

echo "🚀 Starting Secure Heart Attack Prediction System..."
echo "===================================================="

# Change to project directory
cd /home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Start the server using uvicorn (correct way)
echo "🌐 Starting FastAPI server..."
echo "📍 Server will be available at: http://127.0.0.1:8002"
echo "📚 API documentation at: http://127.0.0.1:8002/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

# Use uvicorn to run the app correctly
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload