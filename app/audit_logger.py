"""
Enhanced Logging and Audit System

This module provides comprehensive logging capabilities for the Heart Attack Prediction System,
including audit trails, security events, and model performance tracking.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import wraps
from .config import settings

# Create logs directory
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)

class SecurityAuditLogger:
    """Specialized logger for security and audit events"""
    
    def __init__(self):
        self.setup_loggers()
    
    def setup_loggers(self):
        """Setup different loggers for different purposes"""
        
        # Main application logger
        self.app_logger = logging.getLogger("heart_attack_app")
        self.app_logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Security events logger
        self.security_logger = logging.getLogger("security_audit")
        self.security_logger.setLevel(logging.INFO)
        
        # Model performance logger
        self.model_logger = logging.getLogger("model_performance")
        self.model_logger.setLevel(logging.INFO)
        
        # User activity logger
        self.activity_logger = logging.getLogger("user_activity")
        self.activity_logger.setLevel(logging.INFO)
        
        # Setup formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        
        # Setup file handlers
        if not self.app_logger.handlers:
            # Main app log
            app_handler = logging.FileHandler(settings.log_file)
            app_handler.setFormatter(formatter)
            self.app_logger.addHandler(app_handler)
            
            # Security audit log
            security_handler = logging.FileHandler('logs/security_audit.log')
            security_handler.setFormatter(security_formatter)
            self.security_logger.addHandler(security_handler)
            
            # Model performance log
            model_handler = logging.FileHandler('logs/model_performance.log')
            model_handler.setFormatter(formatter)
            self.model_logger.addHandler(model_handler)
            
            # User activity log
            activity_handler = logging.FileHandler('logs/user_activity.log')
            activity_handler.setFormatter(formatter)
            self.activity_logger.addHandler(activity_handler)
            
            # Console handler for development
            if settings.is_development():
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.app_logger.addHandler(console_handler)
    
    def log_security_event(self, event_type: str, user: str = None, details: Dict[str, Any] = None, success: bool = True):
        """Log security-related events"""
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user": user,
            "success": success,
            "details": details or {}
        }
        
        level = logging.INFO if success else logging.WARNING
        self.security_logger.log(level, json.dumps(event_data))
    
    def log_user_activity(self, user: str, action: str, details: Dict[str, Any] = None):
        """Log user activities"""
        activity_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user,
            "action": action,
            "details": details or {}
        }
        
        self.activity_logger.info(json.dumps(activity_data))
    
    def log_model_prediction(self, user: str, model_name: str, risk_level: str, confidence: float, 
                           patient_data: Dict[str, Any] = None):
        """Log model predictions for performance tracking"""
        prediction_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user,
            "model_name": model_name,
            "risk_level": risk_level,
            "confidence": confidence,
            "patient_data_hash": hash(str(sorted((patient_data or {}).items())))  # Privacy-preserving hash
        }
        
        self.model_logger.info(json.dumps(prediction_data))
    
    def log_data_access(self, user: str, data_type: str, operation: str, record_id: str = None):
        """Log data access events for compliance"""
        access_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user,
            "data_type": data_type,
            "operation": operation,
            "record_id": record_id
        }
        
        self.activity_logger.info(f"DATA_ACCESS: {json.dumps(access_data)}")

# Global audit logger instance
audit_logger = SecurityAuditLogger()

def log_api_call(operation: str):
    """Decorator to log API calls"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful operation
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                audit_logger.app_logger.info(
                    f"API_CALL: {operation} completed successfully in {execution_time:.3f}s"
                )
                
                return result
                
            except Exception as e:
                # Log failed operation
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                audit_logger.app_logger.error(
                    f"API_CALL: {operation} failed after {execution_time:.3f}s - {str(e)}"
                )
                raise
        
        return wrapper
    return decorator

def log_authentication_attempt(email: str, success: bool, failure_reason: str = None):
    """Log authentication attempts"""
    details = {"email": email}
    if not success and failure_reason:
        details["failure_reason"] = failure_reason
    
    audit_logger.log_security_event(
        event_type="authentication_attempt",
        user=email,
        details=details,
        success=success
    )

def log_user_registration(email: str, role: str, success: bool, failure_reason: str = None):
    """Log user registration attempts"""
    details = {"email": email, "role": role}
    if not success and failure_reason:
        details["failure_reason"] = failure_reason
    
    audit_logger.log_security_event(
        event_type="user_registration",
        user=email,
        details=details,
        success=success
    )

def log_prediction_request(user: str, model_result: Dict[str, Any], patient_data: Dict[str, Any]):
    """Log prediction requests with privacy considerations"""
    
    # Log user activity
    audit_logger.log_user_activity(
        user=user,
        action="prediction_request",
        details={
            "model_used": model_result.get("model_used"),
            "risk_level": model_result.get("risk_level"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    # Log model performance
    audit_logger.log_model_prediction(
        user=user,
        model_name=model_result.get("model_used", "unknown"),
        risk_level=model_result.get("risk_level", "unknown"),
        confidence=model_result.get("model_confidence", 0.0),
        patient_data=patient_data
    )

def log_data_validation_error(user: str, validation_errors: List[str]):
    """Log data validation errors"""
    audit_logger.log_security_event(
        event_type="data_validation_error",
        user=user,
        details={"validation_errors": validation_errors},
        success=False
    )

def log_unauthorized_access(user: str = None, endpoint: str = None, reason: str = None):
    """Log unauthorized access attempts"""
    audit_logger.log_security_event(
        event_type="unauthorized_access",
        user=user,
        details={"endpoint": endpoint, "reason": reason},
        success=False
    )

# Setup root logger
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler() if settings.is_development() else logging.NullHandler()
    ]
)