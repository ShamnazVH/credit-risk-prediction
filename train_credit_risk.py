# Fixed training script for credit risk delinquency analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import joblib
import pickle
import os

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Create necessary directories
os.makedirs('models', exist_ok=True)

print("="*80)
print("CREDIT RISK DELINQUENCY ANALYSIS - OPTIMIZED PIPELINE")
print("="*80)

# ============================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================

def load_data():
    """Load dataset from CSV or Excel file"""
    try:
        # Try CSV first
        if os.path.exists("delinquency_prediction_dataset.csv"):
            df = pd.read_csv("delinquency_prediction_dataset.csv")
            print(f"✅ Dataset loaded successfully from CSV: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        # Try Excel
        elif os.path.exists("Delinquency_prediction_dataset.xlsx"):
            df = pd.read_excel("Delinquency_prediction_dataset.xlsx", engine="openpyxl")
            print(f"✅ Dataset loaded successfully from Excel: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        else:
            print("❌ Dataset not found. Creating sample dataset for testing...")
            return create_sample_dataset()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Creating sample dataset for testing...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset for testing if real data is not available"""
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'Age': np.random.randint(18, 80, n_samples),
        'Income': np.random.randint(20000, 200000, n_samples),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Credit_Utilization': np.random.uniform(0, 1, n_samples),
        'Missed_Payments': np.random.randint(0, 10, n_samples),
        'Loan_Balance': np.random.randint(0, 100000, n_samples),
        'Debt_to_Income_Ratio': np.random.uniform(0, 1, n_samples),
        'Account_Tenure': np.random.randint(1, 120, n_samples),
        'Employment_Status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], n_samples),
        'Credit_Card_Type': np.random.choice(['Standard', 'Gold', 'Platinum', 'Premium'], n_samples),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'Month_1': np.random.choice(['On-time', 'Late', 'Missed'], n_samples),
        'Month_2': np.random.choice(['On-time', 'Late', 'Missed'], n_samples),
        'Month_3': np.random.choice(['On-time', 'Late', 'Missed'], n_samples),
        'Month_4': np.random.choice(['On-time', 'Late', 'Missed'], n_samples),
        'Month_5': np.random.choice(['On-time', 'Late', 'Missed'], n_samples),
        'Month_6': np.random.choice(['On-time', 'Late', 'Missed'], n_samples),
    })
    
    # Create target variable based on risk factors
    df['Delinquent_Account'] = (
        (df['Credit_Score'] < 600) | 
        (df['Credit_Utilization'] > 0.7) | 
        (df['Missed_Payments'] > 3) |
        (df['Debt_to_Income_Ratio'] > 0.5)
    ).astype(int)
    
    print(f"✅ Sample dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

df = load_data()
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n📋 First 5 rows:")
print(df.head())

# ============================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

df_clean = df.copy()

def handle_missing_data(df):
    """Handle missing values in the dataset"""
    print("\n🔍 Handling missing values...")
    
    # Numerical columns - mean imputation
    numerical_cols = ['Age', 'Income', 'Credit_Score', 'Credit_Utilization', 
                     'Missed_Payments', 'Loan_Balance', 'Debt_to_Income_Ratio', 'Account_Tenure']
    
    for col in numerical_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna(df[col].mean(), inplace=True)
                print(f"✅ Filled {missing_count} missing values in {col} with mean")
    
    # Categorical columns - mode imputation
    categorical_cols = ['Employment_Status', 'Credit_Card_Type', 'Location']
    
    for col in categorical_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"✅ Filled {missing_count} missing values in {col} with: {mode_val}")
    
    # Payment history columns
    payment_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
    for col in payment_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna('On-time', inplace=True)
                print(f"✅ Filled {missing_count} missing values in {col} with 'On-time'")
    
    return df

df_clean = handle_missing_data(df_clean)

# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Payment behavior encoding
payment_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
payment_mapping = {'On-time': 0, 'Late': 1, 'Missed': 2}

print("\n📊 Encoding payment history...")
for col in payment_cols:
    if col in df_clean.columns:
        df_clean[f'{col}_encoded'] = df_clean[col].map(payment_mapping).fillna(0)
        print(f"✅ Encoded {col}")

# Create payment risk features
if all(f'{col}_encoded' in df_clean.columns for col in payment_cols):
    encoded_cols = [f'{col}_encoded' for col in payment_cols]
    df_clean['Total_Late_Payments'] = df_clean[encoded_cols].apply(lambda x: (x == 1).sum(), axis=1)
    df_clean['Total_Missed_Payments'] = df_clean[encoded_cols].apply(lambda x: (x == 2).sum(), axis=1)
    df_clean['Payment_Risk_Score'] = df_clean['Total_Late_Payments'] + 2 * df_clean['Total_Missed_Payments']
    print("✅ Created payment risk features")

# Risk flags
df_clean['High_Utilization'] = (df_clean['Credit_Utilization'] > 0.7).astype(int)
df_clean['High_DTI'] = (df_clean['Debt_to_Income_Ratio'] > 0.4).astype(int)
df_clean['Low_Credit_Score'] = (df_clean['Credit_Score'] < 580).astype(int)
print("✅ Created risk flag features")

# Encode categorical variables
categorical_cols = ['Employment_Status', 'Credit_Card_Type', 'Location']
label_encoders = {}

print("\n📊 Encoding categorical variables...")
for col in categorical_cols:
    if col in df_clean.columns:
        le = LabelEncoder()
        df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
        print(f"✅ Encoded {col} with {len(le.classes_)} categories: {list(le.classes_)}")

print("\n✅ Feature engineering completed")

# ============================================================
# SECTION 4: PREPARE DATA FOR MODELING
# ============================================================
print("\n" + "="*80)
print("PREPARING DATA FOR MODELING")
print("="*80)

# Check if target variable exists
if 'Delinquent_Account' not in df_clean.columns:
    print("❌ Target variable 'Delinquent_Account' not found!")
    exit()

# Select features for modeling
feature_candidates = [
    'Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments',
    'Loan_Balance', 'Debt_to_Income_Ratio', 'Account_Tenure', 
    'High_Utilization', 'High_DTI', 'Low_Credit_Score', 
    'Payment_Risk_Score', 'Total_Late_Payments', 'Total_Missed_Payments',
    'Employment_Status_encoded', 'Credit_Card_Type_encoded', 'Location_encoded'
]

# Only use features that exist in dataframe
X_features = [f for f in feature_candidates if f in df_clean.columns]
X = df_clean[X_features].copy()
y = df_clean['Delinquent_Account'].copy()

print(f"✅ Using {len(X_features)} features for modeling")
print(f"Features: {X_features}")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Class balance: {y.value_counts(normalize=True)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📈 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# ============================================================
# SECTION 5: TRAIN BEST 3 MODELS WITH HYPERPARAMETER TUNING
# ============================================================
print("\n" + "="*80)
print("TRAINING OPTIMIZED MODELS")
print("="*80)

# Define simplified parameters for faster training
models = {
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 6]
        }
    }
}

# Train and optimize models
results = {}
best_models = {}

for name, config in models.items():
    print(f"\n🏋️ Training and optimizing {name}...")
    
    try:
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=3,  # Reduced from 5 for faster training
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Predictions
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=3, scoring='roc_auc')
        
        # Store results
        results[name] = {
            'model': best_model,
            'accuracy': accuracy,
            'auc': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': best_params
        }
        
        best_models[name] = best_model
        
        print(f"✅ {name} optimized")
        print(f"   Best params: {best_params}")
        print(f"   Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
    except Exception as e:
        print(f"❌ Error training {name}: {e}")

# ============================================================
# SECTION 6: MODEL COMPARISON AND SELECTION
# ============================================================
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

if results:
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'AUC Score': [results[name]['auc'] for name in results.keys()],
        'CV Mean AUC': [results[name]['cv_mean'] for name in results.keys()],
        'CV Std': [results[name]['cv_std'] for name in results.keys()]
    }).sort_values('AUC Score', ascending=False)
    
    print("\n📊 MODEL PERFORMANCE RANKING:")
    print(comparison_df.to_string(index=False))
    
    # Select best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    best_results = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL SELECTED: {best_model_name}")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    print(f"   AUC Score: {best_results['auc']:.4f}")
    print(f"   CV AUC: {best_results['cv_mean']:.4f} (±{best_results['cv_std']:.4f})")
    
    # ============================================================
    # SECTION 7: SAVE MODELS AND ARTIFACTS FOR DEPLOYMENT
    # ============================================================
    print("\n" + "="*80)
    print("SAVING MODELS FOR FLASK DEPLOYMENT")
    print("="*80)
    
    # Save best model
    joblib.dump(best_model, 'models/best_credit_risk_model.pkl')
    print("✅ Saved: models/best_credit_risk_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Saved: models/scaler.pkl")
    
    # Save label encoders
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    print("✅ Saved: models/label_encoders.pkl")
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X_features, f)
    print("✅ Saved: models/feature_names.pkl")
    
    # Save all models
    for name, model in best_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, f'models/{safe_name}_model.pkl')
    print("✅ Saved all individual models")
    
    # Save feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        print("✅ Saved: models/feature_importance.csv")
        
        print("\n📊 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    print(f"\n✅ TRAINING COMPLETED! Best model: {best_model_name}")
else:
    print("❌ No models were successfully trained!")

print("="*80)