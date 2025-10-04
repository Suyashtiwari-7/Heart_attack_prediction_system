from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Import routes with proper path handling
try:
    from app import routes
except ImportError:
    # If running directly, adjust import path
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import routes

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def root():
    """Serve the main HTML interface"""
    return FileResponse("app/static/index.html")

# include routes
app.include_router(routes.router)
