"""
FastAPI Application Entry Point

AI-Generated Image Detection Using Hybrid Forensic Features and Deep Learning
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS
from app.routes import health, analysis
from app.ml.inference import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model on startup."""
    print("[App] Starting up — loading model...")
    engine.load_model()
    print("[App] Model loaded. Server ready.")
    yield
    print("[App] Shutting down.")


app = FastAPI(
    title="AI-Generated Image Detection API",
    description="Detect AI-generated images using hybrid forensic features (ELA, SRM, DCT) and deep learning (EfficientNet-B0).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(analysis.router)


@app.get("/")
async def root():
    return {
        "name": "AI-Generated Image Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze (POST)",
        }
    }
