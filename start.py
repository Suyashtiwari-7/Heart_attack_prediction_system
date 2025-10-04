#!/usr/bin/env python3
"""
Simple startup script for the Heart Attack Prediction System
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🏥 Secure Heart Attack Prediction System")
    print("=" * 50)
    
    # Get the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Check if virtual environment exists
    venv_dir = project_dir / "venv"
    if not venv_dir.exists():
        print("❌ Virtual environment not found. Please create it first:")
        print("   python -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        return
    
    # Check if models exist
    models_dir = project_dir / "models"
    if not models_dir.exists() or not any(models_dir.glob("*.joblib")):
        print("⚠️  ML models not found. Training models...")
        try:
            subprocess.run([str(venv_dir / "bin" / "python"), "train.py"], check=True)
            print("✅ Models trained successfully!")
        except subprocess.CalledProcessError:
            print("❌ Model training failed. The system will use fallback predictions.")
    
    print("🚀 Starting the system...")
    print("📍 URL: http://127.0.0.1:8002")
    print("📚 API Docs: http://127.0.0.1:8002/docs")
    print("🔄 Auto-reload enabled")
    print("-" * 50)
    
    # Start the server
    try:
        python_path = venv_dir / "bin" / "python"
        subprocess.run([
            str(python_path), 
            "-m", "uvicorn", 
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8002",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()