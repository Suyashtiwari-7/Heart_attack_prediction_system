#!/usr/bin/env python3
"""
Debug script to test password hashing directly
"""

import sys
sys.path.append('/home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system')

try:
    from app.security import hash_password, validate_password_strength
    
    test_password = "SecurePass123!"
    
    print("🔧 Testing password functions directly...")
    
    # Test validation
    is_valid, msg = validate_password_strength(test_password)
    print(f"Password validation: {is_valid} - {msg}")
    
    if is_valid:
        # Test hashing
        print("Attempting to hash password...")
        hashed = hash_password(test_password)
        print(f"✅ Password hashed successfully!")
        print(f"Hash: {hashed[:50]}...")
    else:
        print("❌ Password validation failed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()