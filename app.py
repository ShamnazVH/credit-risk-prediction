# Fixed Flask app for credit risk delinquency prediction
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load models and artifacts
def load_artifacts():
    """Load all trained models and preprocessing artifacts"""
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            print("❌ Models directory not found! Please run train_credit_risk.py first.")
            return None, None, None, None
        
        # Load model
        model_path = 'models/best_credit_risk_model.pkl'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None, None, None, None
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # Load scaler
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Scaler loaded")
        
        # Load label encoders
        label_encoders = joblib.load('models/label_encoders.pkl')
        print("✅ Label encoders loaded")
        
        # Load feature names
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = joblib.load(f)
        print(f"✅ Feature names loaded: {len(feature_names)} features")
        
        print("✅ All artifacts loaded successfully")
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

model, scaler, label_encoders, feature_names = load_artifacts()

# Payment mapping
payment_mapping = {'On-time': 0, 'Late': 1, 'Missed': 2}

@app.route('/')
def home():
    """Render the main page"""
    if model is None:
        return """
        <html>
        <body style="font-family: Arial; padding: 50px;">
            <h1>❌ Model Not Loaded</h1>
            <p>Please run <code>python train_credit_risk.py</code> first to train the model.</p>
        </body>
        </html>
        """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for a single customer"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please run train_credit_risk.py first.'
            })
        
        # Get form data
        data = request.form.to_dict()
        
        # Create feature dictionary
        features = {}
        
        # Basic numerical features
        try:
            features['Age'] = float(data.get('age', 0))
            features['Income'] = float(data.get('income', 0))
            features['Credit_Score'] = float(data.get('credit_score', 0))
            features['Credit_Utilization'] = float(data.get('credit_utilization', 0))
            features['Missed_Payments'] = float(data.get('missed_payments', 0))
            features['Loan_Balance'] = float(data.get('loan_balance', 0))
            features['Debt_to_Income_Ratio'] = float(data.get('debt_to_income', 0))
            features['Account_Tenure'] = float(data.get('account_tenure', 0))
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid numeric value: {e}'
            })
        
        # Payment history
        payment_features = {}
        for month in ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']:
            payment_status = data.get(month.lower(), 'On-time')
            payment_features[f'{month}_encoded'] = payment_mapping.get(payment_status, 0)
        
        # Calculate payment risk features
        encoded_payments = list(payment_features.values())
        features['Total_Late_Payments'] = encoded_payments.count(1)
        features['Total_Missed_Payments'] = encoded_payments.count(2)
        features['Payment_Risk_Score'] = features['Total_Late_Payments'] + 2 * features['Total_Missed_Payments']
        
        # Risk flags
        features['High_Utilization'] = 1 if features['Credit_Utilization'] > 0.7 else 0
        features['High_DTI'] = 1 if features['Debt_to_Income_Ratio'] > 0.4 else 0
        features['Low_Credit_Score'] = 1 if features['Credit_Score'] < 580 else 0
        
        # Encode categorical variables
        employment_status = data.get('employment_status', 'Employed')
        credit_card_type = data.get('credit_card_type', 'Standard')
        location = data.get('location', 'Urban')
        
        if label_encoders:
            try:
                # Handle unknown categories gracefully
                if employment_status in label_encoders['Employment_Status'].classes_:
                    features['Employment_Status_encoded'] = label_encoders['Employment_Status'].transform([employment_status])[0]
                else:
                    features['Employment_Status_encoded'] = 0
                    
                if credit_card_type in label_encoders['Credit_Card_Type'].classes_:
                    features['Credit_Card_Type_encoded'] = label_encoders['Credit_Card_Type'].transform([credit_card_type])[0]
                else:
                    features['Credit_Card_Type_encoded'] = 0
                    
                if location in label_encoders['Location'].classes_:
                    features['Location_encoded'] = label_encoders['Location'].transform([location])[0]
                else:
                    features['Location_encoded'] = 0
            except Exception as e:
                print(f"Warning: Error encoding categorical variables: {e}")
                features['Employment_Status_encoded'] = 0
                features['Credit_Card_Type_encoded'] = 0
                features['Location_encoded'] = 0
        else:
            features['Employment_Status_encoded'] = 0
            features['Credit_Card_Type_encoded'] = 0
            features['Location_encoded'] = 0
        
        # Create feature array in correct order
        feature_array = []
        for feature in feature_names:
            feature_array.append(features.get(feature, 0))
        
        # Convert to numpy array and reshape
        feature_array = np.array(feature_array).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_array_scaled)[0]
        probability = model.predict_proba(feature_array_scaled)[0]
        
        # Interpret results
        result = "High Risk - Potential Delinquency" if prediction == 1 else "Low Risk - Good Standing"
        risk_level = "High" if prediction == 1 else "Low"
        
        # Get the probability for the predicted class
        delinquency_prob = probability[1]
        confidence = delinquency_prob if prediction == 1 else (1 - delinquency_prob)
        
        # Risk factors
        risk_factors = []
        if features['High_Utilization'] == 1:
            risk_factors.append(f"High Credit Utilization ({features['Credit_Utilization']*100:.1f}% > 70%)")
        if features['High_DTI'] == 1:
            risk_factors.append(f"High Debt-to-Income Ratio ({features['Debt_to_Income_Ratio']*100:.1f}% > 40%)")
        if features['Low_Credit_Score'] == 1:
            risk_factors.append(f"Low Credit Score ({features['Credit_Score']:.0f} < 580)")
        if features['Total_Missed_Payments'] > 0:
            risk_factors.append(f"Missed Payments ({int(features['Total_Missed_Payments'])} times in last 6 months)")
        if features['Total_Late_Payments'] > 0:
            risk_factors.append(f"Late Payments ({int(features['Total_Late_Payments'])} times in last 6 months)")
        if features['Payment_Risk_Score'] > 3:
            risk_factors.append(f"High Payment Risk Score ({int(features['Payment_Risk_Score'])})")
        
        response = {
            'success': True,
            'prediction': int(prediction),
            'probability': round(float(delinquency_prob), 4),
            'result': result,
            'risk_level': risk_level,
            'confidence': round(float(confidence), 4),
            'risk_factors': risk_factors,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in prediction: {error_msg}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Process batch predictions from uploaded file"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please run train_credit_risk.py first.'
            })
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format. Please use CSV or XLSX.'})
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully. Batch processing feature coming soon!',
            'records_received': len(df),
            'columns': list(df.columns)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            })
        
        info = {
            'success': True,
            'model_type': type(model).__name__,
            'feature_count': len(feature_names),
            'features': feature_names
        }
        
        # Try to load feature importance if available
        if os.path.exists('models/feature_importance.csv'):
            feature_importance = pd.read_csv('models/feature_importance.csv')
            info['top_features'] = feature_importance.head(10).to_dict('records')
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("="*80)
    print("🚀 CREDIT RISK PREDICTION API")
    print("="*80)
    
    if model is not None:
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Features: {len(feature_names)}")
        print(f"✅ Feature list: {feature_names}")
        print("🌐 Server starting on http://localhost:5000")
        print("="*80)
    else:
        print("⚠️  WARNING: Model not loaded!")
        print("📋 Please run: python train_credit_risk.py")
        print("="*80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)