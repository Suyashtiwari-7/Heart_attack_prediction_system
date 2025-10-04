#!/usr/bin/env python3
"""
Test the prediction functionality to see the updated response
"""

import requests
import json

def test_prediction():
    base_url = "http://127.0.0.1:8002/api"
    
    print("🧪 Testing Updated Prediction System")
    print("=" * 50)
    
    # First register and login
    print("1. Registering test user...")
    reg_data = {
        'email': 'testuser2@example.com',
        'password': 'SecurePass123!',
        'role': 'Patient'
    }
    
    try:
        response = requests.post(f'{base_url}/register', json=reg_data, timeout=5)
        if response.status_code == 200:
            print("   ✅ Registration successful!")
        else:
            print(f"   ⚠️ Registration response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Registration error: {e}")
        return
    
    # Login
    print("\n2. Logging in...")
    login_data = {
        'email': 'testuser2@example.com',
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
    
    # Test prediction
    print("\n3. Testing prediction...")
    prediction_data = {
        'age': 55,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'cholesterol': 250,
        'heart_rate': 85
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
            print(f"   Feature Importance Keys: {list(result.get('feature_importance', {}).keys())}")
        else:
            print(f"   ❌ Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Test Complete!")
    print(f"🌐 Web interface: http://127.0.0.1:8002")

if __name__ == "__main__":
    test_prediction()