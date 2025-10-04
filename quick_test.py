#!/usr/bin/env python3
"""
Quick test for the API functionality
"""

import requests
import json
import time

def main():
    # Wait a moment for server to be ready
    time.sleep(1)
    
    base_url = "http://127.0.0.1:8002/api"
    
    print("🧪 Testing Heart Attack Prediction System API")
    print("=" * 50)
    
    # Test 1: Registration
    print("\n1. Testing Registration...")
    reg_data = {
        'email': 'testuser@example.com',
        'password': 'SecurePass123!',
        'role': 'Patient'
    }
    
    try:
        response = requests.post(f'{base_url}/register', json=reg_data, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        if response.status_code == 200:
            print("   ✅ Registration successful!")
        else:
            print("   ❌ Registration failed!")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Server not running on port 8002")
        return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Login
    print("\n2. Testing Login...")
    login_data = {
        'email': 'testuser@example.com',
        'password': 'SecurePass123!'
    }
    
    try:
        response = requests.post(f'{base_url}/login', json=login_data, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        if response.status_code == 200:
            print("   ✅ Login successful!")
        else:
            print("   ❌ Login failed!")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing Complete!")

if __name__ == "__main__":
    main()