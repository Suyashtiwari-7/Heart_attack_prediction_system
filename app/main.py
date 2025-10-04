"""
Main FastAPI application for Secure Heart Attack Prediction System
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add the parent directory to Python path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now import routes
from app import routes

# Create FastAPI app
app = FastAPI(
    title="Secure Heart Attack Prediction System",
    description="ML-powered heart attack risk prediction with secure authentication",
    version="1.0.0"
)

# Mount static files (adjust path based on execution context)
static_path = current_dir / "static"
if not static_path.exists():
    static_path = Path("app/static")

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    """Serve the main HTML interface"""
    html_path = static_path / "index.html"
    if not html_path.exists():
        html_path = Path("app/static/index.html")
    return FileResponse(str(html_path))

# Include API routes
app.include_router(routes.router)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Secure Heart Attack Prediction System...")
    print("📍 Server will be available at: http://127.0.0.1:8002")
    print("📚 API documentation at: http://127.0.0.1:8002/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8002,
        reload=True,
        log_level="info"
    )
