import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import router as api_router
from configurations.config import settings

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="UltraSafe Data Processing and Analysis System with RAG and Multi-Agent Workflows",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)
os.makedirs("exports", exist_ok=True)
os.makedirs("db", exist_ok=True)




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
