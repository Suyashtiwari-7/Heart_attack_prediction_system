
"""
Security Module for Heart Attack Prediction System

Handles password hashing, verification, and other security utilities.
"""

import bcrypt
import hashlib
import logging

logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with proper length handling.
    bcrypt has a maximum input length of 72 bytes.
    """
    try:
        # Handle password length limitation (bcrypt max 72 bytes)
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Pre-hash long passwords with SHA256 to ensure they fit within bcrypt's limit
            password_bytes = hashlib.sha256(password_bytes).hexdigest().encode('utf-8')
        
        # Generate salt and hash the password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        return hashed.decode('utf-8')
        
    except Exception as e:
        logger.error(f"Password hashing failed: {str(e)}")
        raise ValueError(f"Password hashing failed: {str(e)}")

def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a password against its hash with proper length handling.
    """
    try:
        # Handle password length limitation (same logic as hashing)
        password_bytes = plain.encode('utf-8')
        if len(password_bytes) > 72:
            password_bytes = hashlib.sha256(password_bytes).hexdigest().encode('utf-8')
        
        hashed_bytes = hashed.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
        
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False

def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password strength and return (is_valid, error_message).
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if len(password) > 128:
        return False, "Password must be less than 128 characters long"
    
    # Check for at least one uppercase, lowercase, digit, and special character
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if not (has_upper and has_lower and has_digit and has_special):
        return False, "Password must contain at least one uppercase letter, lowercase letter, digit, and special character"
    
    return True, "Password is valid"
