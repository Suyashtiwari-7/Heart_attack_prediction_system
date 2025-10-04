# 🏥 Secure Heart Attack Prediction System

A complete machine learning system for predicting heart attack risk with secure authentication and interactive visualizations.

## 🚀 Quick Start

### Easiest Way to Run:
```bash
cd /home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system
python start.py
```

### Alternative Methods:
```bash
# Method 1: Direct run
python app/main.py

# Method 2: Using uvicorn
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload
```

## 🌐 Access Your System

- **Web Interface**: http://127.0.0.1:8002
- **API Documentation**: http://127.0.0.1:8002/docs

## ✨ What You Can Do

1. **Register** a new account with secure password
2. **Login** to access the prediction system
3. **Enter health data** (age, blood pressure, cholesterol, heart rate)
4. **Get risk assessment** with personalized prevention tips
5. **View health trends** with interactive charts

## 🏗️ Clean Project Structure

```
├── app/
│   ├── main.py              # Main application
│   ├── routes.py            # API endpoints
│   ├── security.py          # Authentication
│   ├── ml_models.py         # ML predictions
│   └── static/index.html    # Web interface
├── models/best_model.joblib # Trained ML model (92.6% confidence)
├── data/heart_disease_uci.csv # Training data
├── start.py                 # Easy startup script
└── requirements.txt         # Dependencies
```

## 🎯 System Features

- ✅ **ML Model**: CatBoost with 92.6% confidence, AUC 0.9261
- ✅ **Security**: JWT authentication, bcrypt password hashing
- ✅ **Web UI**: Interactive charts, real-time predictions
- ✅ **Validation**: Comprehensive input validation
- ✅ **Prevention**: Personalized health recommendations

## 🛠️ Troubleshooting

**Can't access website?**
```bash
# Kill any running servers
pkill -f uvicorn
# Restart
python start.py
```

**Import errors?**
```bash
# Make sure you're in the right directory
cd /home/suyashtiwari/Documents/CODE/secure_heart_attack_prediction_system
# Activate virtual environment
source venv/bin/activate
```

---

**🎉 Your system is ready! Visit: http://127.0.0.1:8002**