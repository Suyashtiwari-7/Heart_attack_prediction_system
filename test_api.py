#!/usr/bin/env python3
"""
Test script for registration and login functionality
"""

import requests
import json

def test_api():
    base_url = "http://127.0.0.1:8002/api"
    
    print("🧪 Testing Heart Attack Prediction System API")
    print("=" * 50)
    
    # Test 1: Registration with valid password
    print("\n1. Testing Registration...")
    reg_data = {
        'email': 'testuser@example.com',
        'password': 'SecurePass123!',
        'role': 'Patient'
    }
    
    try:
        response = requests.post(f'{base_url}/register', json=reg_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Registration successful: {response.json()}")
        else:
            print(f"   ❌ Registration failed: {response.json()}")
    except Exception as e:
        print(f"   ❌ Network error: {e}")
    
    # Test 2: Login with correct credentials
    print("\n2. Testing Login...")
    login_data = {
        'email': 'testuser@example.com',
        'password': 'SecurePass123!'
    }
    
    try:
        response = requests.post(f'{base_url}/login', json=login_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Login successful!")
            print(f"   Token type: {result.get('token_type')}")
            print(f"   Token (first 20 chars): {result.get('access_token', '')[:20]}...")
            return result.get('access_token')
        else:
            print(f"   ❌ Login failed: {response.json()}")
    except Exception as e:
        print(f"   ❌ Network error: {e}")
    
    # Test 3: Weak password registration
    print("\n3. Testing Weak Password Registration...")
    weak_reg_data = {
        'email': 'weakpass@example.com',
        'password': '123',
        'role': 'Patient'
    }
    
    try:
        response = requests.post(f'{base_url}/register', json=weak_reg_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 400:
            print(f"   ✅ Correctly rejected weak password: {response.json()}")
        else:
            print(f"   ❌ Should have rejected weak password: {response.json()}")
    except Exception as e:
        print(f"   ❌ Network error: {e}")
    
    # Test 4: Invalid login
    print("\n4. Testing Invalid Login...")
    invalid_login_data = {
        'email': 'testuser@example.com',
        'password': 'WrongPassword123!'
    }
    
    try:
        response = requests.post(f'{base_url}/login', json=invalid_login_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 401:
            print(f"   ✅ Correctly rejected invalid credentials: {response.json()}")
        else:
            print(f"   ❌ Should have rejected invalid credentials: {response.json()}")
    except Exception as e:
        print(f"   ❌ Network error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Testing Complete!")
    print("\n💡 Now try the web interface at: http://127.0.0.1:8002")
    print("   Use these test credentials:")
    print("   Email: testuser@example.com")
    print("   Password: SecurePass123!")

if __name__ == "__main__":
    test_api()