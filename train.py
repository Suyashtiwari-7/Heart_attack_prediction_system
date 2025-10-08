
"""
Enhanced Training Script for Heart Attack Prediction System

This script:
- Loads the UCI heart disease dataset
- Performs comprehensive preprocessing
- Trains XGBoost, LightGBM and CatBoost models
- Evaluates and compares all models using multiple metrics
- Creates visualization comparing model performance
- Saves the best performing model with full comparison report
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

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
    print("\n" + "="*60)
    print("🏆 MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Create comprehensive comparison DataFrame
    comparison_data = []
    
    for model_name, model in models.items():
        if model_name == 'ensemble':
            continue  # Skip ensemble for individual model comparison
            
        # Get predictions
        if model_name == 'catboost':
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        comparison_data.append({
            'Model': model_name.upper(),
            'Accuracy': accuracy,
            'AUC Score': auc,
            'CV Mean': None  # Will fill this later
        })
    
    # Convert to DataFrame for easy display
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AUC Score', ascending=False)
    
    # Print comparison table
    print("\n📊 PERFORMANCE METRICS COMPARISON:")
    print("-" * 50)
    print(f"{'Model':<15} {'Accuracy':<12} {'AUC Score':<12}")
    print("-" * 50)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:<15} {row['Accuracy']:<12.4f} {row['AUC Score']:<12.4f}")
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model'].lower()
    best_accuracy = comparison_df.iloc[0]['Accuracy']
    best_auc = comparison_df.iloc[0]['AUC Score']
    
    print("-" * 50)
    print(f"🥇 BEST MODEL: {best_model_name.upper()}")
    print(f"   📈 Accuracy: {best_accuracy:.4f}")
    print(f"   📊 AUC Score: {best_auc:.4f}")
    print("-" * 50)
    
    # Create visualization
    print("\n📈 Creating performance visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(comparison_df['Model'], comparison_df['Accuracy'], 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars1, comparison_df['Accuracy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight best model
    best_idx = comparison_df['Model'].tolist().index(best_model_name.upper())
    bars1[best_idx].set_color('#FFD93D')
    bars1[best_idx].set_edgecolor('#FF6B35')
    bars1[best_idx].set_linewidth(3)
    
    # AUC Score comparison
    bars2 = ax2.bar(comparison_df['Model'], comparison_df['AUC Score'], 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Model AUC Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUC Score', fontsize=12)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars2, comparison_df['AUC Score']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight best model
    bars2[best_idx].set_color('#FFD93D')
    bars2[best_idx].set_edgecolor('#FF6B35')
    bars2[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(MODEL_DIR, "model_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"� Visualization saved to: {plot_path}")
    
    # Show the plot briefly
    plt.show(block=False)
    plt.pause(2)  # Display for 2 seconds
    plt.close()
    
    # Detailed classification reports for all models
    print(f"\n📋 DETAILED CLASSIFICATION REPORTS:")
    print("="*60)
    
    for model_name, model in models.items():
        if model_name == 'ensemble':
            continue
            
        print(f"\n🔍 {model_name.upper()} Classification Report:")
        print("-" * 40)
        
        # Get predictions
        if model_name == 'catboost':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)
        
        print(classification_report(y_test, y_pred))
    
    # Save the best model
    best_model = models[best_model_name]
    best_score = model_scores[best_model_name]
    
    print(f"\n💾 SAVING BEST MODEL: {best_model_name.upper()}")
    
    # Highlight XGBoost if it's the best
    if best_model_name == 'xgboost':
        print("🌟" * 20)
        print("🚀 XGBOOST IS THE CHAMPION! 🚀")
        print("   XGBoost has achieved the highest performance")
        print("   and will be used as the primary model!")
        print("🌟" * 20)
    
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'model_score': best_score,
        'use_scaled_data': best_model_name != 'catboost',
        'comparison_data': comparison_df.to_dict('records'),
        'all_models': models  # Save all models for ensemble use
    }
    
    joblib.dump(model_artifacts, os.path.join(MODEL_DIR, "best_model.joblib"))
    print(f"✅ Model artifacts saved to: {os.path.join(MODEL_DIR, 'best_model.joblib')}")
    
    # Save comparison results
    comparison_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)
    print(f"📊 Comparison results saved to: {os.path.join(MODEL_DIR, 'model_comparison.csv')}")
    
    # Feature importance analysis
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS ({best_model_name.upper()}):")
    print("-" * 50)
    
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🏆 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        # Save feature importance
        importance_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(10)
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top 10 Feature Importances - {best_model_name.upper()} Model', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        feature_plot_path = os.path.join(MODEL_DIR, "feature_importance.png")
        plt.savefig(feature_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Feature importance plot saved to: {feature_plot_path}")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    
else:
    print("❌ No models were successfully trained!")

print("\n✅ Training completed!")
print("🔧 You can now use the trained model in your FastAPI application.")
