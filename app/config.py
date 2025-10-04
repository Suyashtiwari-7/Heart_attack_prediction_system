"""
Configuration Management Module

This module handles all configuration settings including environment variables,
security settings, and database connections for the Heart Attack Prediction System.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
import secrets

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Security Settings
    secret_key: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", secrets.token_urlsafe(32)),
        description="Secret key for JWT token generation"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration time")
    
    # Database Configuration (Oracle)
    oracle_host: str = Field(default="localhost", description="Oracle database host")
    oracle_port: int = Field(default=1521, description="Oracle database port")
    oracle_service_name: str = Field(default="ORCL", description="Oracle service name")
    oracle_username: str = Field(default="heart_attack_user", description="Oracle username")
    oracle_password: str = Field(default="", description="Oracle password")
    oracle_dsn: Optional[str] = Field(default=None, description="Oracle DSN string")
    oracle_tns_admin: Optional[str] = Field(default=None, description="Oracle TNS admin path")
    database_role: str = Field(default="APP_ROLE", description="Database role")
    
    # Application Settings
    app_host: str = Field(default="0.0.0.0", description="Application host")
    app_port: int = Field(default=8000, description="Application port")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="production", description="Environment")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/app.log", description="Log file path")
    
    # Model Configuration
    model_path: str = Field(default="models/best_model.joblib", description="ML model path")
    model_update_interval: int = Field(default=86400, description="Model update interval in seconds")
    
    # Security Headers
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
        description="CORS allowed origins"
    )
    enable_https: bool = Field(default=True, description="Enable HTTPS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def oracle_connection_string(self) -> str:
        """Generate Oracle connection string"""
        if self.oracle_dsn:
            return self.oracle_dsn
        return f"{self.oracle_username}/{self.oracle_password}@{self.oracle_host}:{self.oracle_port}/{self.oracle_service_name}"
    
    @property
    def database_url(self) -> str:
        """Generate database URL for SQLAlchemy"""
        return f"oracle+cx_oracle://{self.oracle_connection_string}"
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() in ["development", "dev", "local"]
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() in ["production", "prod"]

# Global settings instance
settings = Settings()

# Validate critical settings
if not settings.secret_key or settings.secret_key == "your-super-secure-secret-key-change-this-in-production":
    if settings.is_production():
        raise ValueError("SECRET_KEY must be set to a secure value in production!")
    else:
        print("⚠️  Warning: Using default/insecure SECRET_KEY in development mode")

if not settings.oracle_password and settings.is_production():
    print("⚠️  Warning: Oracle password not set. Database features may not work.")

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(settings.log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)