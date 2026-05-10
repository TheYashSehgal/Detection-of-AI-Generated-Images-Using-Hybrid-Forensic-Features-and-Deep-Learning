"""Health check endpoint."""
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.ml.inference import engine

router = APIRouter(tags=["Health"])

@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=engine.is_loaded,
        version="1.0.0"
    )
