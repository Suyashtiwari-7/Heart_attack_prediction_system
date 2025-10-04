
"""
Enhanced Training Script for Heart Attack Prediction System

This script:
- Loads the UCI heart disease dataset
- Performs comprehensive preprocessing
- Trains XGBoost, LightGBM and CatBoost models
- Evaluates and saves the best performing model
- Saves preprocessing components for production use
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("🏥 Heart Attack Prediction System - Model Training")
print("=" * 50)

# Load the heart disease dataset
csv_path = os.path.join(DATA_DIR, "heart_disease_uci.csv")
if not os.path.exists(csv_path):
    print("❌ Dataset not found! Please ensure heart_disease_uci.csv is in the data/ directory.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)
print(f"📊 Loaded dataset with shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

# Data preprocessing for UCI heart disease dataset
def preprocess_data(df):
    """Comprehensive preprocessing for heart disease dataset"""
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Create binary target (any heart disease = 1, no disease = 0)
    data['target'] = (data['num'] > 0).astype(int)
    
    # Handle categorical variables
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    
    # Create label encoders
    label_encoders = {}
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    
    # Select features for prediction
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                      'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Ensure all feature columns exist
    available_features = [col for col in feature_columns if col in data.columns]
    
    X = data[available_features]
    y = data['target']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"✅ Features selected: {available_features}")
    print(f"📈 Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, label_encoders, available_features

# Preprocess the data
X, y, label_encoders, feature_names = preprocess_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"🔄 Training set size: {X_train.shape[0]}")
print(f"🔄 Test set size: {X_test.shape[0]}")

# Initialize models dictionary
models = {}
model_scores = {}

# Train XGBoost
try:
    import xgboost as xgb
    print("\n🚀 Training XGBoost...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Cross-validation
    cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"   📊 Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    models['xgboost'] = xgb_model
    model_scores['xgboost'] = auc_score
    print(f"   ✅ Test AUC: {auc_score:.4f}")
    
except ImportError:
    print("   ⚠️  XGBoost not available")
except Exception as e:
    print(f"   ❌ Error training XGBoost: {e}")

# Train LightGBM
try:
    import lightgbm as lgb
    print("\n🚀 Training LightGBM...")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(lgb_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"   📊 Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    lgb_model.fit(X_train_scaled, y_train)
    y_pred_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    models['lightgbm'] = lgb_model
    model_scores['lightgbm'] = auc_score
    print(f"   ✅ Test AUC: {auc_score:.4f}")
    
except ImportError:
    print("   ⚠️  LightGBM not available")
except Exception as e:
    print(f"   ❌ Error training LightGBM: {e}")

# Train CatBoost
try:
    from catboost import CatBoostClassifier
    print("\n🚀 Training CatBoost...")
    
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    
    # Cross-validation
    cv_scores = cross_val_score(cat_model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"   📊 Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set (CatBoost can handle non-scaled data)
    cat_model.fit(X_train, y_train)
    y_pred_proba = cat_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    models['catboost'] = cat_model
    model_scores['catboost'] = auc_score
    print(f"   ✅ Test AUC: {auc_score:.4f}")
    
except ImportError:
    print("   ⚠️  CatBoost not available")
except Exception as e:
    print(f"   ❌ Error training CatBoost: {e}")

# Create ensemble model if multiple models are available
if len(models) > 1:
    print("\n🤖 Creating Ensemble Model...")
    
    # Prepare estimators for voting classifier
    estimators = []
    for name, model in models.items():
        if name == 'catboost':
            # CatBoost needs unscaled data, so we'll use a wrapper
            estimators.append((name, model))
        else:
            estimators.append((name, model))
    
    # For now, use soft voting with the scaled models only
    scaled_estimators = [(name, model) for name, model in estimators if name != 'catboost']
    
    if len(scaled_estimators) > 1:
        ensemble = VotingClassifier(estimators=scaled_estimators, voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        
        y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
        ensemble_auc = roc_auc_score(y_test, y_pred_proba)
        
        models['ensemble'] = ensemble
        model_scores['ensemble'] = ensemble_auc
        print(f"   ✅ Ensemble AUC: {ensemble_auc:.4f}")

# Select and save the best model
if model_scores:
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = models[best_model_name]
    best_score = model_scores[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_score:.4f})")
    
    # Save the best model and preprocessing components
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'model_score': best_score,
        'use_scaled_data': best_model_name != 'catboost'
    }
    
    joblib.dump(model_artifacts, os.path.join(MODEL_DIR, "best_model.joblib"))
    print(f"💾 Model artifacts saved to: {os.path.join(MODEL_DIR, 'best_model.joblib')}")
    
    # Generate classification report
    if best_model_name == 'catboost':
        y_pred = best_model.predict(X_test)
    else:
        y_pred = best_model.predict(X_test_scaled)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n🎯 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 10 Feature Importances:")
        print(importance_df.head(10).to_string(index=False))
        
        # Save feature importance
        importance_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    
else:
    print("❌ No models were successfully trained!")

print("\n✅ Training completed!")
print("🔧 You can now use the trained model in your FastAPI application.")
