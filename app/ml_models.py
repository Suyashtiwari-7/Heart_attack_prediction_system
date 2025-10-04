"""
ML Model Integration Module

This module handles loading and using the trained machine learning models
for heart attack risk prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class HeartAttackPredictor:
    """Heart Attack Risk Prediction using trained ML models"""
    
    def __init__(self, model_path: str = None):
        self.model_artifacts = None
        self.is_loaded = False
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the trained model and preprocessing components"""
        try:
            if os.path.exists(model_path):
                self.model_artifacts = joblib.load(model_path)
                self.is_loaded = True
                logger.info(f"✅ Loaded {self.model_artifacts['model_name']} model "
                           f"(AUC: {self.model_artifacts['model_score']:.4f})")
            else:
                logger.warning(f"⚠️  Model file not found: {model_path}")
                logger.warning("Using fallback simple prediction method")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.warning("Using fallback simple prediction method")
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict heart attack risk for a patient
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Dictionary with prediction results
        """
        
        if not self.is_loaded or self.model_artifacts is None:
            return self._fallback_prediction(patient_data)
        
        try:
            # Prepare input data
            input_features = self._prepare_input(patient_data)
            
            # Get model components
            model = self.model_artifacts['model']
            scaler = self.model_artifacts['scaler']
            use_scaled_data = self.model_artifacts['use_scaled_data']
            
            # Make prediction
            if use_scaled_data:
                input_scaled = scaler.transform([input_features])
                risk_probability = model.predict_proba(input_scaled)[0][1]
            else:
                # For CatBoost, use unscaled data
                input_df = pd.DataFrame([input_features], columns=self.model_artifacts['feature_names'])
                risk_probability = model.predict_proba(input_df)[0][1]
            
            # Convert to risk level
            risk_level = self._probability_to_risk_level(risk_probability)
            
            # Get feature contributions (if available)
            feature_importance = self._get_feature_importance(input_features)
            
            return {
                "risk_probability": round(float(risk_probability), 4),
                "risk_level": risk_level,
                "model_used": self.model_artifacts['model_name'],
                "model_confidence": self.model_artifacts['model_score'],
                "feature_importance": feature_importance,
                "prediction_details": {
                    "input_features": dict(zip(self.model_artifacts['feature_names'], input_features)),
                    "risk_factors": self._identify_risk_factors(patient_data, risk_probability)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return self._fallback_prediction(patient_data)
    
    def _prepare_input(self, patient_data: Dict[str, Any]) -> List[float]:
        """Prepare input features in the correct format for the model"""
        
        # Map input fields to model features
        feature_mapping = {
            'age': patient_data.get('age', 50),
            'sex': 1 if patient_data.get('sex', 'Male').lower() == 'male' else 0,
            'cp': patient_data.get('chest_pain_type', 0),  # Default typical angina
            'trestbps': patient_data.get('systolic_bp', 120),
            'chol': patient_data.get('cholesterol', 200),
            'fbs': 1 if patient_data.get('fasting_blood_sugar', 120) > 120 else 0,
            'restecg': patient_data.get('rest_ecg', 0),  # Default normal
            'thalch': patient_data.get('max_heart_rate', 
                                        # If max_heart_rate not provided, estimate from heart_rate
                                        # Max HR is typically higher than resting HR
                                        max(patient_data.get('heart_rate', 150) * 1.5, 60) if patient_data.get('heart_rate') else 150),
            'exang': patient_data.get('exercise_angina', 0),  # Default no
            'oldpeak': patient_data.get('st_depression', 0),
            'slope': patient_data.get('st_slope', 1),  # Default upsloping
            'ca': patient_data.get('ca_vessels', 0),  # Default 0 vessels
            'thal': patient_data.get('thalassemia', 2)  # Default normal
        }
        
        # Get features in the correct order
        features = []
        for feature_name in self.model_artifacts['feature_names']:
            features.append(feature_mapping.get(feature_name, 0))
        
        return features
    
    def _probability_to_risk_level(self, probability: float) -> str:
        """Convert probability to risk level category"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        else:
            return "High"
    
    def _get_feature_importance(self, input_features: List[float]) -> Dict[str, float]:
        """Get feature importance for the current prediction"""
        try:
            model = self.model_artifacts['model']
            if hasattr(model, 'feature_importances_'):
                importance_dict = {}
                for i, feature_name in enumerate(self.model_artifacts['feature_names']):
                    importance_dict[feature_name] = round(float(model.feature_importances_[i]), 4)
                return importance_dict
        except:
            pass
        return {}
    
    def _identify_risk_factors(self, patient_data: Dict[str, Any], risk_prob: float) -> List[str]:
        """Identify key risk factors for the patient"""
        risk_factors = []
        
        age = patient_data.get('age', 0)
        systolic_bp = patient_data.get('systolic_bp', 0)
        cholesterol = patient_data.get('cholesterol', 0)
        
        if age > 65:
            risk_factors.append("Advanced age (>65 years)")
        elif age > 55:
            risk_factors.append("Moderate age risk (55-65 years)")
        
        if systolic_bp > 140:
            risk_factors.append("High blood pressure (>140 mmHg)")
        elif systolic_bp > 130:
            risk_factors.append("Elevated blood pressure (130-140 mmHg)")
        
        if cholesterol > 240:
            risk_factors.append("High cholesterol (>240 mg/dL)")
        elif cholesterol > 200:
            risk_factors.append("Borderline high cholesterol (200-240 mg/dL)")
        
        # Add more risk factor identification based on other parameters
        if patient_data.get('exercise_angina', 0) == 1:
            risk_factors.append("Exercise-induced angina")
        
        if patient_data.get('chest_pain_type', 0) == 0:  # Assuming 0 is typical angina
            risk_factors.append("Typical angina symptoms")
        
        if not risk_factors:
            if risk_prob > 0.5:
                risk_factors.append("Combination of multiple moderate risk factors")
            else:
                risk_factors.append("Low overall risk profile")
        
        return risk_factors
    
    def _fallback_prediction(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback simple prediction when ML model is not available"""
        logger.warning("Using fallback prediction method")
        
        # Simple risk calculation (original logic)
        age = patient_data.get('age', 50)
        systolic_bp = patient_data.get('systolic_bp', 120)
        cholesterol = patient_data.get('cholesterol', 200)
        heart_rate = patient_data.get('heart_rate', 70)
        
        score = 0
        score += (age / 100) * 0.3
        score += (systolic_bp / 200) * 0.25
        score += (cholesterol / 300) * 0.25
        score += (heart_rate / 200) * 0.2
        
        risk_probability = min(max(score, 0), 1)
        risk_level = self._probability_to_risk_level(risk_probability)
        
        return {
            "risk_probability": round(float(risk_probability), 4),
            "risk_level": risk_level,
            "model_used": "fallback_simple",
            "model_confidence": 0.7,  # Lower confidence for simple model
            "feature_importance": {},
            "prediction_details": {
                "input_features": {
                    "age": age,
                    "systolic_bp": systolic_bp,
                    "cholesterol": cholesterol,
                    "heart_rate": heart_rate
                },
                "risk_factors": self._identify_risk_factors(patient_data, risk_probability)
            }
        }

# Global predictor instance
predictor = HeartAttackPredictor()