# Credit Risk Delinquency Prediction System

AI-powered credit risk prediction system using machine learning to assess the likelihood of customer delinquency.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
```bash
cd credit-risk-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**
   - Place your dataset file in the project root directory
   - Supported formats: CSV or Excel (.xlsx)
   - File should be named: `delinquency_prediction_dataset.csv` or `Delinquency_prediction_dataset.xlsx`
   - If you don't have a dataset, the training script will create a sample dataset automatically

4. **Train the model**
```bash
python train_credit_risk.py
```

This will:
- Load and preprocess the data
- Engineer features
- Train multiple models (XGBoost, Random Forest, Gradient Boosting)
- Select the best performing model
- Save all artifacts to the `models/` directory

5. **Run the Flask application**
```bash
python app.py
```

6. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
credit-risk-prediction/
├── app.py                          # Flask web application
├── train_credit_risk.py            # Model training script
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Web interface
├── models/                         # Generated model artifacts
│   ├── best_credit_risk_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   └── feature_importance.csv
└── delinquency_prediction_dataset.csv  # Your dataset
```

## 📊 Required Dataset Columns

Your dataset should include the following columns:

### Numerical Features
- `Age`: Customer age (18-100)
- `Income`: Annual income in dollars
- `Credit_Score`: Credit score (300-850)
- `Credit_Utilization`: Credit utilization ratio (0-1)
- `Missed_Payments`: Number of missed payments in the last year
- `Loan_Balance`: Current loan balance in dollars
- `Debt_to_Income_Ratio`: Debt-to-income ratio (0-1)
- `Account_Tenure`: Account age in months

### Categorical Features
- `Employment_Status`: Employed, Self-Employed, Unemployed, Retired
- `Credit_Card_Type`: Standard, Gold, Platinum, Premium
- `Location`: Urban, Suburban, Rural

### Payment History (Last 6 Months)
- `Month_1` through `Month_6`: On-time, Late, or Missed

### Target Variable
- `Delinquent_Account`: 0 (Good Standing) or 1 (Delinquent)

## 🔧 Troubleshooting

### Issue: "Model not loaded" error
**Solution**: Run `python train_credit_risk.py` first to train and save the model.

### Issue: "Dataset not found" error
**Solution**: 
- Ensure your dataset file is in the project root directory
- Check the file name matches: `delinquency_prediction_dataset.csv` or `Delinquency_prediction_dataset.xlsx`
- Or let the script generate a sample dataset automatically

### Issue: Import errors
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Port 5000 already in use
**Solution**: Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## 📈 Model Performance

The system trains three models and automatically selects the best:
- **XGBoost**: Gradient boosting with extreme optimization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble method

Models are evaluated using:
- Accuracy
- AUC-ROC Score
- Cross-validation scores

## 🎯 Features

### Web Interface
- Interactive form for single predictions
- Real-time risk assessment
- Confidence scores
- Risk factor identification
- Visual risk indicators

### Model Capabilities
- Handles missing data automatically
- Feature engineering and scaling
- Hyperparameter optimization
- Categorical encoding
- Payment behavior analysis

## 📝 API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Make single prediction
- `POST /batch_predict` - Upload CSV/Excel for batch predictions
- `GET /model_info` - Get model information
- `GET /health` - Health check endpoint

## 🔐 Security Notes

This is a demonstration application. For production use:
- Add authentication and authorization
- Implement rate limiting
- Add input validation and sanitization
- Use HTTPS
- Store sensitive data securely
- Add logging and monitoring

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your dataset format matches the requirements
4. Check console output for detailed error messages