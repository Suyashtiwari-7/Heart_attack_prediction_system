#!/usr/bin/env python3
"""
Test the fixed prediction functionality
"""

import requests
import json

def test_fixed_prediction():
    base_url = "http://127.0.0.1:8002/api"
    
    print("🧪 Testing Fixed Prediction System")
    print("=" * 50)
    
    # Register a new user
    print("1. Registering test user...")
    reg_data = {
        'email': 'fixedtest@example.com',
        'password': 'SecurePass123!',
        'role': 'Patient'
    }
    
    try:
        response = requests.post(f'{base_url}/register', json=reg_data, timeout=5)
        print(f"   Registration: {response.status_code}")
    except Exception as e:
        print(f"   Registration error: {e}")
        return
    
    # Login
    print("\n2. Logging in...")
    login_data = {
        'email': 'fixedtest@example.com',
        'password': 'SecurePass123!'
    }
    
    try:
        response = requests.post(f'{base_url}/login', json=login_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            token = result.get('access_token')
            print("   ✅ Login successful!")
        else:
            print(f"   ❌ Login failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Login error: {e}")
        return
    
    # Test prediction with valid data
    print("\n3. Testing prediction with valid data...")
    prediction_data = {
        'age': 55,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'cholesterol': 250,
        'heart_rate': 75  # Valid heart rate
    }
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.post(f'{base_url}/predict', json=prediction_data, headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful!")
            print(f"   Risk Level: {result.get('risk_level')}")
            print(f"   Risk Probability: {result.get('risk_probability', 0) * 100:.1f}%")
            print(f"   Model Used: {result.get('model_used')}")
            print(f"   Model Confidence: {result.get('model_confidence', 0) * 100:.1f}%")
        else:
            result = response.json()
            print(f"   ❌ Prediction failed: {result}")
            
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    # Test with low heart rate (edge case)
    print("\n4. Testing prediction with low heart rate...")
    prediction_data_low_hr = {
        'age': 30,
        'systolic_bp': 110,
        'diastolic_bp': 70,
        'cholesterol': 180,
        'heart_rate': 45  # Low but valid heart rate
    }
    
    try:
        response = requests.post(f'{base_url}/predict', json=prediction_data_low_hr, headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Low heart rate prediction successful!")
            print(f"   Risk Level: {result.get('risk_level')}")
        else:
            result = response.json()
            print(f"   ❌ Low heart rate prediction failed: {result}")
            
    except Exception as e:
        print(f"   ❌ Low heart rate prediction error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing Complete!")
    print("🌐 Web interface: http://127.0.0.1:8002")

if __name__ == "__main__":
    test_fixed_prediction()