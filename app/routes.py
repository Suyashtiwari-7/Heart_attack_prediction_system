from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field, conint, confloat
from .security import hash_password, verify_password, validate_password_strength
from .ml_models import predictor
from .config import settings
from .audit_logger import (
    log_authentication_attempt, log_user_registration, 
    log_prediction_request, log_unauthorized_access
)
import logging
from typing import Optional, List, Annotated
from datetime import datetime, timedelta
import jwt

router = APIRouter(prefix="/api")

# Simple in-memory 'users' demo (replace with DB later)
USERS = {}

# Configuration from settings
SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

# ------------------ MODELS ------------------
class RegisterModel(BaseModel):
    email: str
    password: str
    role: str = "Patient"  # Patient / Doctor / Admin

class LoginModel(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PredictModel(BaseModel):
    age: Annotated[int, Field(ge=0, le=120, description="Patient age in years")]
    systolic_bp: Annotated[float, Field(ge=50, le=300, description="Systolic blood pressure in mmHg")]
    diastolic_bp: Annotated[float, Field(ge=30, le=200, description="Diastolic blood pressure in mmHg")]
    cholesterol: Annotated[float, Field(ge=50, le=600, description="Cholesterol level in mg/dL")]
    heart_rate: Annotated[float, Field(ge=30, le=220, description="Heart rate in bpm")]
    
    # Additional optional parameters for enhanced prediction
    sex: Optional[str] = Field("Male", description="Patient sex (Male/Female)")
    chest_pain_type: Optional[Annotated[int, Field(ge=0, le=3)]] = Field(0, description="Chest pain type (0-3)")
    fasting_blood_sugar: Optional[Annotated[float, Field(ge=50, le=400)]] = Field(100, description="Fasting blood sugar in mg/dL")
    rest_ecg: Optional[Annotated[int, Field(ge=0, le=2)]] = Field(0, description="Resting ECG results (0-2)")
    max_heart_rate: Optional[Annotated[float, Field(ge=60, le=220)]] = Field(150, description="Maximum heart rate achieved")
    exercise_angina: Optional[Annotated[int, Field(ge=0, le=1)]] = Field(0, description="Exercise induced angina (0=No, 1=Yes)")
    st_depression: Optional[Annotated[float, Field(ge=0, le=10)]] = Field(0, description="ST depression induced by exercise")
    st_slope: Optional[Annotated[int, Field(ge=0, le=2)]] = Field(1, description="Slope of peak exercise ST segment")
    ca_vessels: Optional[Annotated[int, Field(ge=0, le=4)]] = Field(0, description="Number of major vessels colored by fluoroscopy")
    thalassemia: Optional[Annotated[int, Field(ge=0, le=3)]] = Field(2, description="Thalassemia type (0-3)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "systolic_bp": 130,
                "diastolic_bp": 85,
                "cholesterol": 220,
                "heart_rate": 75,
                "sex": "Male",
                "chest_pain_type": 1,
                "fasting_blood_sugar": 110,
                "rest_ecg": 0,
                "max_heart_rate": 160,
                "exercise_angina": 0,
                "st_depression": 1.2,
                "st_slope": 1,
                "ca_vessels": 0,
                "thalassemia": 2
            }
        }

class PredictionResponse(BaseModel):
    user: str
    risk_probability: float
    risk_level: str
    model_used: str
    model_confidence: float
    feature_importance: dict
    prediction_details: dict
    timestamp: datetime

# ------------------ HELPERS ------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_from_header(authorization: str = Header(None)) -> str:
    """Extract and validate user from Authorization header with audit logging"""
    try:
        if not authorization:
            log_unauthorized_access(None, "No authorization header", "Missing Authorization header")
            raise HTTPException(status_code=401, detail="Authorization header missing")
        
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            log_unauthorized_access(None, "Invalid auth scheme", f"Invalid scheme: {scheme}")
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in USERS:
            log_unauthorized_access(email, "Invalid token", "User not found or token invalid")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        return email
        
    except ValueError:
        log_unauthorized_access(None, "Invalid header format", "Cannot parse Authorization header")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    except jwt.ExpiredSignatureError:
        log_unauthorized_access(None, "Expired token", "JWT token has expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError as e:
        log_unauthorized_access(None, "JWT error", str(e))
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in USERS:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ------------------ ROUTES ------------------
@router.post("/register")
def register(payload: RegisterModel):
    """Register a new user with comprehensive validation and audit logging"""
    try:
        # Validate email format
        if not payload.email or "@" not in payload.email:
            log_user_registration(payload.email, payload.role, False, "Invalid email format")
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Check if user already exists
        if payload.email in USERS:
            log_user_registration(payload.email, payload.role, False, "User already exists")
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Validate password strength
        is_valid, error_message = validate_password_strength(payload.password)
        if not is_valid:
            log_user_registration(payload.email, payload.role, False, f"Password validation failed: {error_message}")
            raise HTTPException(status_code=400, detail=f"Password validation failed: {error_message}")
        
        # Validate role
        valid_roles = ["Patient", "Doctor", "Admin"]
        if payload.role not in valid_roles:
            log_user_registration(payload.email, payload.role, False, f"Invalid role: {payload.role}")
            raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
        
        # Hash password
        try:
            hashed = hash_password(payload.password)
        except ValueError as e:
            log_user_registration(payload.email, payload.role, False, f"Password hashing failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Password processing failed")
        
        # Store user
        USERS[payload.email] = {"password": hashed, "role": payload.role}
        
        # Log successful registration
        log_user_registration(payload.email, payload.role, True)
        logging.info(f"Registered user {payload.email} with role {payload.role}")
        
        return {"status": "registered", "email": payload.email, "role": payload.role}
        
    except HTTPException:
        raise
    except Exception as e:
        log_user_registration(payload.email if hasattr(payload, 'email') else "unknown", 
                             payload.role if hasattr(payload, 'role') else "unknown", 
                             False, str(e))
        logging.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginModel):
    """Authenticate user with comprehensive audit logging"""
    try:
        user = USERS.get(payload.email)
        if not user or not verify_password(payload.password, user["password"]):
            log_authentication_attempt(payload.email, False, "Invalid credentials")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        access_token = create_access_token(data={"sub": payload.email})
        
        # Log successful authentication
        log_authentication_attempt(payload.email, True)
        logging.info(f"Successful login for user {payload.email}")
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException:
        raise
    except Exception as e:
        log_authentication_attempt(payload.email, False, str(e))
        logging.error(f"Login failed for {payload.email}: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictModel, user: str = Depends(get_current_user_from_header)):
    """Enhanced heart attack risk prediction using trained ML models with audit logging"""
    
    try:
        # Convert Pydantic model to dictionary for the predictor
        patient_data = payload.dict()
        
        # Use the ML predictor
        prediction_result = predictor.predict_risk(patient_data)
        
        # Prepare response
        response = {
            "user": user,
            "risk_probability": prediction_result['risk_probability'],
            "risk_level": prediction_result['risk_level'],
            "model_used": prediction_result['model_used'],
            "model_confidence": prediction_result['model_confidence'],
            "feature_importance": prediction_result['feature_importance'],
            "prediction_details": prediction_result['prediction_details'],
            "timestamp": datetime.utcnow()
        }
        
        # Log the prediction with audit trail
        log_prediction_request(user, prediction_result, patient_data)
        
        logging.info(
            f"Prediction by {user}: {prediction_result['risk_level']} risk "
            f"(probability: {prediction_result['risk_probability']:.4f}, "
            f"model: {prediction_result['model_used']})"
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Error during prediction for user {user}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )
