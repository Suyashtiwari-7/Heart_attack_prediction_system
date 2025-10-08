"""
ENHANCED Training Script for Heart Attack Prediction System
🎯 FOCUSED ON MAXIMUM ACCURACY

Advanced techniques for increasing model accuracy:
1. Feature engineering and polynomial features
2. Hyperparameter optimization using GridSearch
3. Ensemble methods with stacking
4. Advanced cross-validation
5. Feature selection and dimensionality reduction
6. Data augmentation techniques
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    print("✅ All ML libraries loaded successfully")
except ImportError as e:
    print(f"❌ Missing ML library: {e}")
    print("🔧 Installing required packages...")
    os.system("pip install xgboost catboost lightgbm")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("🎯 ENHANCED Heart Attack Prediction System - Maximum Accuracy Training")
print("=" * 70)

# Load the heart disease dataset
csv_path = os.path.join(DATA_DIR, "heart_disease_uci.csv")
if not os.path.exists(csv_path):
    print("❌ Dataset not found! Please ensure heart_disease_uci.csv is in the data/ directory.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)
print(f"📊 Loaded dataset with shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

def advanced_preprocessing(df):
    """Advanced preprocessing with feature engineering"""
    print("🔧 Advanced preprocessing...")
    
    # Make a copy
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.dropna()
    
    # Feature engineering - create interaction features
    print("⚙️  Creating interaction features...")
    
    # Age-based features
    df_processed['age_group'] = pd.cut(df_processed['age'], bins=[0, 35, 50, 65, 100], 
                                      labels=['young', 'middle', 'senior', 'elderly'])
    df_processed['age_group_encoded'] = df_processed['age_group'].astype('category').cat.codes
    
    # Blood pressure features
    if 'trestbps' in df_processed.columns:
        df_processed['bp_category'] = pd.cut(df_processed['trestbps'], 
                                           bins=[0, 120, 140, 180, 300], 
                                           labels=['normal', 'elevated', 'high', 'very_high'])
        df_processed['bp_category_encoded'] = df_processed['bp_category'].astype('category').cat.codes
    
    # Cholesterol features
    if 'chol' in df_processed.columns:
        df_processed['chol_category'] = pd.cut(df_processed['chol'], 
                                             bins=[0, 200, 240, 300, 600], 
                                             labels=['normal', 'borderline', 'high', 'very_high'])
        df_processed['chol_category_encoded'] = df_processed['chol_category'].astype('category').cat.codes
    
    # Heart rate features
    if 'thalch' in df_processed.columns:
        df_processed['hr_reserve'] = 220 - df_processed['age'] - df_processed['thalch']
        df_processed['hr_percentage'] = (df_processed['thalch'] / (220 - df_processed['age'])) * 100
    
    # Interaction features
    if 'age' in df_processed.columns and 'chol' in df_processed.columns:
        df_processed['age_chol_interaction'] = df_processed['age'] * df_processed['chol'] / 1000
    
    if 'trestbps' in df_processed.columns and 'chol' in df_processed.columns:
        df_processed['bp_chol_ratio'] = df_processed['trestbps'] / (df_processed['chol'] + 1)
    
    # Remove non-numeric categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    df_processed = df_processed.drop(columns=categorical_cols)
    
    print(f"✅ Features after engineering: {df_processed.shape[1]}")
    return df_processed

def optimize_hyperparameters(X_train, y_train, cv_folds=5):
    """Hyperparameter optimization for all models"""
    print("🔍 Optimizing hyperparameters...")
    
    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # LightGBM hyperparameters
    lgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'feature_fraction': [0.8, 0.9, 1.0]
    }
    
    # CatBoost hyperparameters
    cat_params = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5]
    }
    
    optimized_models = {}
    
    # Optimize XGBoost
    print("   🎯 Optimizing XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_grid = RandomizedSearchCV(xgb_model, xgb_params, cv=cv_folds, 
                                 scoring='roc_auc', n_iter=20, random_state=42)
    xgb_grid.fit(X_train, y_train)
    optimized_models['XGBoost'] = xgb_grid.best_estimator_
    print(f"      Best XGBoost AUC: {xgb_grid.best_score_:.4f}")
    
    # Optimize LightGBM
    print("   🎯 Optimizing LightGBM...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_grid = RandomizedSearchCV(lgb_model, lgb_params, cv=cv_folds, 
                                 scoring='roc_auc', n_iter=20, random_state=42)
    lgb_grid.fit(X_train, y_train)
    optimized_models['LightGBM'] = lgb_grid.best_estimator_
    print(f"      Best LightGBM AUC: {lgb_grid.best_score_:.4f}")
    
    # Optimize CatBoost
    print("   🎯 Optimizing CatBoost...")
    cat_model = CatBoostClassifier(random_state=42, verbose=0)
    cat_grid = RandomizedSearchCV(cat_model, cat_params, cv=cv_folds, 
                                 scoring='roc_auc', n_iter=20, random_state=42)
    cat_grid.fit(X_train, y_train)
    optimized_models['CatBoost'] = cat_grid.best_estimator_
    print(f"      Best CatBoost AUC: {cat_grid.best_score_:.4f}")
    
    return optimized_models

def create_ensemble_model(optimized_models, X_train, y_train):
    """Create ensemble models for maximum accuracy"""
    print("🤝 Creating ensemble models...")
    
    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in optimized_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    
    # Stacking Classifier
    from sklearn.linear_model import LogisticRegression
    stacking_clf = StackingClassifier(
        estimators=[(name, model) for name, model in optimized_models.items()],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    
    return {
        'VotingClassifier': voting_clf,
        'StackingClassifier': stacking_clf,
        **optimized_models
    }

def evaluate_models_comprehensive(models, X_test, y_test, X_train, y_train):
    """Comprehensive model evaluation"""
    print("📊 Comprehensive model evaluation...")
    
    results = []
    
    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'AUC Score': auc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        print(f"   {name}:")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      AUC: {auc:.4f}")
        print(f"      CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
    
    return pd.DataFrame(results)

# Main training process
def main():
    # Advanced preprocessing
    df_processed = advanced_preprocessing(df)
    
    # Prepare target variable
    target_col = None
    possible_targets = ['target', 'heart_disease', 'num', 'class']
    for col in possible_targets:
        if col in df_processed.columns:
            target_col = col
            break
    
    if target_col is None:
        # Assume last column is target
        target_col = df_processed.columns[-1]
        print(f"⚠️  Assuming '{target_col}' is the target variable")
    
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Convert target to binary if needed
    if y.nunique() > 2:
        y = (y > 0).astype(int)
    
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    # Handle class imbalance with SMOTE
    print("⚖️  Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"   After SMOTE: {pd.Series(y_balanced).value_counts().to_dict()}")
    
    # Feature selection
    print("🎯 Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k=min(15, X_balanced.shape[1]))
    X_selected = selector.fit_transform(X_balanced, y_balanced)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"   Selected features: {len(selected_features)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # Scale features
    print("📏 Scaling features...")
    scaler = RobustScaler()  # More robust to outliers
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter optimization
    optimized_models = optimize_hyperparameters(X_train_scaled, y_train)
    
    # Create ensemble models
    all_models = create_ensemble_model(optimized_models, X_train_scaled, y_train)
    
    # Comprehensive evaluation
    results_df = evaluate_models_comprehensive(all_models, X_test_scaled, y_test, X_train_scaled, y_train)
    
    # Display results
    print("\n🏆 FINAL RESULTS - ACCURACY MAXIMIZATION")
    print("=" * 70)
    print(results_df.round(4).to_string(index=False))
    
    # Find best model
    best_model_name = results_df.loc[results_df['AUC Score'].idxmax(), 'Model']
    best_model = all_models[best_model_name]
    best_auc = results_df.loc[results_df['AUC Score'].idxmax(), 'AUC Score']
    
    print(f"\n🎉 BEST MODEL: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Save the best model
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'feature_selector': selector,
        'selected_features': selected_features,
        'feature_names': selected_features,
        'model_name': best_model_name.lower(),
        'model_score': best_auc,
        'use_scaled_data': True,
        'comparison_data': results_df.to_dict('records'),
        'training_enhanced': True,
        'features_engineered': True,
        'hyperparameters_optimized': True
    }
    
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(model_artifacts, model_path)
    print(f"💾 Enhanced model saved to: {model_path}")
    
    # Create accuracy comparison chart
    plt.figure(figsize=(12, 8))
    
    # AUC Comparison
    plt.subplot(2, 2, 1)
    models_sorted = results_df.sort_values('AUC Score', ascending=True)
    colors = ['gold' if name == best_model_name else 'skyblue' for name in models_sorted['Model']]
    plt.barh(models_sorted['Model'], models_sorted['AUC Score'], color=colors)
    plt.xlabel('AUC Score')
    plt.title('🎯 Model AUC Comparison (Higher = Better)')
    plt.xlim(0, 1)
    
    # Accuracy Comparison
    plt.subplot(2, 2, 2)
    models_sorted = results_df.sort_values('Accuracy', ascending=True)
    colors = ['gold' if name == best_model_name else 'lightcoral' for name in models_sorted['Model']]
    plt.barh(models_sorted['Model'], models_sorted['Accuracy'], color=colors)
    plt.xlabel('Accuracy')
    plt.title('🎯 Model Accuracy Comparison')
    plt.xlim(0, 1)
    
    # F1 Score Comparison
    plt.subplot(2, 2, 3)
    models_sorted = results_df.sort_values('F1 Score', ascending=True)
    colors = ['gold' if name == best_model_name else 'lightgreen' for name in models_sorted['Model']]
    plt.barh(models_sorted['Model'], models_sorted['F1 Score'], color=colors)
    plt.xlabel('F1 Score')
    plt.title('🎯 Model F1 Score Comparison')
    plt.xlim(0, 1)
    
    # Cross-validation comparison
    plt.subplot(2, 2, 4)
    models_sorted = results_df.sort_values('CV Mean', ascending=True)
    colors = ['gold' if name == best_model_name else 'plum' for name in models_sorted['Model']]
    plt.barh(models_sorted['Model'], models_sorted['CV Mean'], color=colors)
    plt.xlabel('CV AUC Score')
    plt.title('🎯 Cross-Validation AUC Comparison')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    chart_path = os.path.join(MODEL_DIR, "enhanced_model_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Enhanced comparison chart saved to: {chart_path}")
    
    # Save detailed results
    results_path = os.path.join(MODEL_DIR, "enhanced_model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"📄 Detailed results saved to: {results_path}")
    
    print("\n🎉 ENHANCED TRAINING COMPLETE!")
    print(f"🏆 Best performing model: {best_model_name}")
    print(f"🎯 Maximum AUC achieved: {best_auc:.4f}")
    print(f"⚡ Features used: {len(selected_features)}")
    print(f"🤖 Model enhancements: Feature Engineering ✓, Hyperparameter Optimization ✓, Ensemble Methods ✓")

if __name__ == "__main__":
    main()