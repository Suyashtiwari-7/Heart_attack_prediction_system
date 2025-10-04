#!/usr/bin/env python3
"""
Debug script to test password length
"""

test_password = "SecurePass123!"
print(f"Password: '{test_password}'")
print(f"Length in chars: {len(test_password)}")
print(f"Length in bytes: {len(test_password.encode('utf-8'))}")
print(f"Is > 72 bytes? {len(test_password.encode('utf-8')) > 72}")

# Test the bcrypt issue with a different approach
import bcrypt

# Try direct bcrypt
try:
    print("\nTesting direct bcrypt...")
    hashed = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt())
    print(f"✅ Direct bcrypt works: {hashed}")
except Exception as e:
    print(f"❌ Direct bcrypt failed: {e}")