# app_final.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import keras

app = Flask(__name__)

class CreditRiskPredictor:
    def __init__(self, model_path='best_credit_risk_model.joblib'):
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            # Load model artifacts
            artifacts = joblib.load(self.model_path)
            
            self.model = artifacts['model']
            self.model_name = artifacts['model_name']
            self.features = artifacts['features']
            self.encoders = artifacts['encoders']
            self.scaler = artifacts.get('scaler', None)
            
            print(f"✓ Model loaded successfully: {self.model_name}")
            print(f"✓ Features: {len(self.features)}")
            print(f"✓ Model type: {type(self.model).__name__}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please run the training script first!")
            raise e
    
    def preprocess_input(self, input_data):
        """Preprocess input data matching training pipeline"""
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Handle payment history
        payment_columns = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
        payment_mapping = {'On-time': 0, 'Late': 1, 'Missed': 2}
        
        for col in payment_columns:
            if col in df.columns:
                df[col] = df[col].map(payment_mapping)
                df[col] = df[col].fillna(0)
        
        # Create engineered features (EXACTLY like training)
        df['Total_Late_Payments'] = df[payment_columns].apply(
            lambda x: (x == 1).sum(), axis=1)
        df['Total_Missed_Payments'] = df[payment_columns].apply(
            lambda x: (x == 2).sum(), axis=1)
        df['Payment_Consistency_Score'] = df[payment_columns].apply(
            lambda x: (x == 0).sum() / len(x), axis=1)
        
        # Additional engineered features
        df['Credit_Utilization_Risk'] = df['Credit_Utilization'] * df['Missed_Payments']
        df['Income_to_Balance_Ratio'] = df['Income'] / (df['Loan_Balance'] + 1)
        
        # Encode categorical variables
        categorical_cols = ['Employment_Status', 'Credit_Card_Type', 'Location']
        for col in categorical_cols:
            if col in df.columns and col in self.encoders:
                try:
                    df[col] = self.encoders[col].transform([str(input_data[col])])[0]
                except ValueError:
                    df[col] = 0  # Default for unseen categories
        
        # Select features in exact same order as training
        processed_data = df[self.features]
        
        return processed_data
    
    def predict_risk(self, input_data):
        """Predict credit risk"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Scale if scaler exists
            if self.scaler:
                processed_data = self.scaler.transform(processed_data)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(processed_data)[0, 1]
            else:
                probability = self.model.predict(processed_data)[0, 0]
            
            # Determine risk category
            if probability < 0.3:
                risk_category = "Low Risk"
                risk_color = "#28a745"
                recommendation = "✅ Approve - Standard terms"
            elif probability < 0.7:
                risk_category = "Medium Risk" 
                risk_color = "#ffc107"
                recommendation = "⚠️ Review - Enhanced monitoring"
            else:
                risk_category = "High Risk"
                risk_color = "#dc3545"
                recommendation = "❌ Decline - High default probability"
            
            # Risk factors analysis
            risk_factors = self.analyze_risk_factors(input_data)
            
            return {
                'probability': round(probability, 4),
                'risk_percentage': round(probability * 100, 2),
                'risk_category': risk_category,
                'risk_color': risk_color,
                'recommendation': recommendation,
                'risk_factors': risk_factors,
                'model_used': self.model_name,
                'confidence': self.calculate_confidence(probability),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'error': str(e)}
    
    def analyze_risk_factors(self, input_data):
        """Analyze risk factors"""
        risk_factors = []
        
        if input_data.get('Credit_Utilization', 0) > 0.7:
            risk_factors.append("High credit utilization (>70%)")
        
        if input_data.get('Missed_Payments', 0) > 3:
            risk_factors.append(f"Multiple missed payments ({input_data['Missed_Payments']})")
        
        if input_data.get('Debt_to_Income_Ratio', 0) > 0.4:
            risk_factors.append(f"High debt-to-income ratio ({input_data['Debt_to_Income_Ratio']:.2f})")
        
        if input_data.get('Credit_Score', 0) < 600:
            risk_factors.append(f"Low credit score ({input_data['Credit_Score']})")
        
        if not risk_factors:
            risk_factors.append("No major risk factors identified")
        
        return risk_factors
    
    def calculate_confidence(self, probability):
        """Calculate confidence level"""
        if probability < 0.2 or probability > 0.8:
            return "High"
        elif probability < 0.3 or probability > 0.7:
            return "Medium"
        else:
            return "Moderate"

# Initialize predictor
try:
    predictor = CreditRiskPredictor()
    models_loaded = True
    print("🎯 Credit Risk Predictor initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize predictor: {e}")
    models_loaded = False

@app.route('/')
def home():
    return render_template('index.html', 
                         model_status="Loaded" if models_loaded else "Not Loaded",
                         model_name=predictor.model_name if models_loaded else "None")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not models_loaded:
            return jsonify({'error': 'Models not loaded'})
        
        # Get input data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Prepare input
        input_data = {
            'Age': float(data.get('Age', 0)),
            'Income': float(data.get('Income', 0)),
            'Credit_Score': float(data.get('Credit_Score', 0)),
            'Credit_Utilization': float(data.get('Credit_Utilization', 0)),
            'Missed_Payments': int(data.get('Missed_Payments', 0)),
            'Loan_Balance': float(data.get('Loan_Balance', 0)),
            'Debt_to_Income_Ratio': float(data.get('Debt_to_Income_Ratio', 0)),
            'Employment_Status': data.get('Employment_Status', 'Employed'),
            'Account_Tenure': int(data.get('Account_Tenure', 0)),
            'Credit_Card_Type': data.get('Credit_Card_Type', 'Standard'),
            'Location': data.get('Location', 'New York'),
            'Month_1': data.get('Month_1', 'On-time'),
            'Month_2': data.get('Month_2', 'On-time'),
            'Month_3': data.get('Month_3', 'On-time'),
            'Month_4': data.get('Month_4', 'On-time'),
            'Month_5': data.get('Month_5', 'On-time'),
            'Month_6': data.get('Month_6', 'On-time')
        }
        
        # Make prediction
        result = predictor.predict_risk(input_data)
        
        if 'error' in result:
            return jsonify({'error': result['error']})
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'model_name': predictor.model_name if models_loaded else 'None'
    })

if __name__ == '__main__':
    print("\n🚀 Starting Credit Risk Prediction Server")
    print("🌐 http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)