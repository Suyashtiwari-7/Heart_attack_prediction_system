# Secure Heart Attack Prediction System

**Overview**
This project implements a secure system to predict heart attack risk using gradient boosting models (XGBoost, LightGBM, CatBoost). It includes:
- Backend API (FastAPI) with authentication & role-based access placeholders
- Training script (`train.py`) demonstrating how to train XGBoost/LightGBM/CatBoost
- Oracle DB connection placeholder and notes for TDE/encryption
- Frontend sample (HTML + Chart.js) for results visualization
- Input validation utilities and logging example

Dataset archive unpacked into data/ directory.

## How to use
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Inspect `data/` for the provided dataset.
3. To train models (if you have ML libraries installed), run:
```
python train.py
```
4. Start API:
```
uvicorn app.main:app --reload
```

## Files
- `app/` FastAPI backend
- `train.py` ML training script
- `data/` dataset extracted from provided archive
- `requirements.txt` required Python packages (placeholders)
- `oracle_notes.md` notes on configuring Oracle TDE and secure connection

