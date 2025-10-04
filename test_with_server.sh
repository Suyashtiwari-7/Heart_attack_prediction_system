#!/bin/bash

# Start server in background
cd /home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system
/home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 &

# Wait for server to start
sleep 5

# Run tests
/home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system/venv/bin/python /home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system/quick_test.py

# Kill the server
pkill -f "uvicorn app.main:app"