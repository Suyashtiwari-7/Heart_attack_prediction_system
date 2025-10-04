#!/bin/bash

# Start server in background
cd /home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system
/home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 &

# Wait for server to start
sleep 4

# Test prediction
/home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system/venv/bin/python -c "
import requests
import json

# Test prediction with proper login
base_url = 'http://127.0.0.1:8002/api'

# Register first
reg_data = {'email': 'testuser3@example.com', 'password': 'SecurePass123!', 'role': 'Patient'}
response = requests.post(f'{base_url}/register', json=reg_data)
print(f'Registration: {response.status_code}')

# Login 
login_data = {'email': 'testuser3@example.com', 'password': 'SecurePass123!'}
response = requests.post(f'{base_url}/login', json=login_data)
if response.status_code == 200:
    token = response.json()['access_token']
    print('✅ Login successful')
    
    # Test prediction
    headers = {'Authorization': f'Bearer {token}'}
    pred_data = {
        'age': 55,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'cholesterol': 250,
        'heart_rate': 85
    }
    
    response = requests.post(f'{base_url}/predict', json=pred_data, headers=headers)
    print(f'Prediction status: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'✅ Prediction successful: {result[\"risk_level\"]} ({result[\"risk_probability\"]*100:.1f}%)')
    else:
        print(f'❌ Prediction failed: {response.text}')
else:
    print(f'❌ Login failed: {response.status_code}')
"

# Kill the server
pkill -f "uvicorn app.main:app"