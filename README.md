# 🫀 Heart Attack Prediction System
The **Heart Attack Prediction System** is a Machine Learning–based web application that predicts the risk of a heart attack based on user health parameters such as age, cholesterol, blood pressure, chest pain type, and more.

The system integrates:
- **Oracle Database** for secure data storage.
- **Gradient Boosting Models (XGBoost, LightGBM, CatBoost)** for high-accuracy predictions.
- **Chart.js** for interactive visualization of user statistics and prediction insights.

This project demonstrates end-to-end integration of **Data Science, Machine Learning, and Web Development** for a practical healthcare use case.

---

## ⚙️ Tech Stack
| Layer | Technology Used |
|-------|------------------|
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **Backend** | Flask (Python) |
| **Database** | Oracle Database |
| **Machine Learning** | XGBoost, LightGBM, CatBoost |
| **Libraries** | pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, matplotlib, seaborn |

---

## 🧩 Features
- 🩺 Predicts heart attack risk using ML models.
- 📊 Displays results through **Chart.js** visualizations.
- 💾 Stores patient data securely in **Oracle Database**.
- ⚡ Gradient boosting ensemble for superior accuracy.
- 🌐 Simple, clean, and responsive interface.

---

## 📚 Dataset
The dataset used is derived from **UCI Heart Disease Dataset** containing patient health metrics:
- `age` – Patient’s age  
- `sex` – Gender (1 = male, 0 = female)  
- `cp` – Chest pain type  
- `trestbps` – Resting blood pressure (mm Hg)  
- `chol` – Serum cholesterol (mg/dl)  
- `fbs` – Fasting blood sugar > 120 mg/dl  
- `restecg` – Resting electrocardiographic results  
- `thalach` – Maximum heart rate achieved  
- `exang` – Exercise induced angina  
- `oldpeak` – ST depression induced by exercise  
- `slope` – Slope of the peak exercise ST segment  
- `ca` – Number of major vessels (0–3) colored by fluoroscopy  
- `thal` – Thalassemia type  
- `target` – 1 = likely heart attack, 0 = unlikely

---

## 🧮 Model Training

### 1️⃣ Data Preprocessing
- Handled missing values, encoded categorical features.
- Performed standard scaling and normalization.
- Split dataset: `80%` training, `20%` testing.

### 2️⃣ Models Used
| Model | Description |
|--------|--------------|
| **XGBoost** | Gradient boosting on decision trees – high speed and performance |
| **LightGBM** | Optimized for speed with large datasets |
| **CatBoost** | Handles categorical data automatically |

### 3️⃣ Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC Curve.
- Ensemble voting of top models for final prediction.

---

## 🧰 Installation & Setup

### 🔧 Prerequisites
- Python 3.8 or above  
- Oracle Database (with user and table created)  
- Installed dependencies

### 📦 Clone the Repository
```bash
git clone https://github.com/your-username/heart-attack-prediction.git
cd heart-attack-prediction
```

📥 Install Required Libraries
--------------------------------
```bash
pip install -r requirements.txt
```

📊 Visualization (Chart.js)
--------------------------------
- Pie charts show the proportion of predicted risk levels.
- Bar graphs visualize top features influencing prediction.
- Line charts show dataset trends (e.g., cholesterol vs age).

Example output:
High Risk: 68%
Medium Risk: 22%
Low Risk: 10%

🧠 Sample Prediction Flow
--------------------------------
1️⃣ User enters their health data on the web form.  
2️⃣ The system preprocesses inputs → sends to trained ensemble model.  
3️⃣ Model predicts probability of heart attack.  
4️⃣ Result is stored in Oracle Database and displayed visually via Chart.js.

📁 Project Structure
--------------------------------
```bash
heart-attack-prediction/
│
├── static/
│   ├── css/
│   ├── js/ (Chart.js)
│
├── templates/
│   ├── index.html
│   ├── result.html
│
├── models/
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── catboost_model.pkl
│
├── app.py
├── config.py
├── requirements.txt
├── train_model.py
└── README.md
```

📈 Model Performance
```bash
--------------------------------
Model        Accuracy    ROC-AUC
--------------------------------
XGBoost      92.4%       0.94
LightGBM     91.8%       0.93
CatBoost     93.1%       0.95
--------------------------------
Final ensemble achieved 94.2% accuracy and 0.96 ROC-AUC.
```


🔒 Security & Data Handling
--------------------------------
- Sensitive user data is stored securely in Oracle DB with credentials hidden in environment variables.
- Input validation on both frontend and backend.
- No personal data is shared externally.

📅 Future Improvements
--------------------------------
- Add real-time sensor/IoT integration (BP, heart rate monitors).
- Deploy to AWS/Render with containerization (Docker).
- Add authentication and user dashboards.

👨‍💻 Author
--------------------------------
Suyash Tiwari


📜 License
--------------------------------
This project is licensed under the MIT License — free to use and modify.

