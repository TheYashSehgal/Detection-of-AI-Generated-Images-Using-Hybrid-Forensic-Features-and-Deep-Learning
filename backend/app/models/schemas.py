"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class AnalysisResponse(BaseModel):
    """Response schema for image analysis endpoint."""
    
    # Prediction
    prediction: str = Field(..., description="'real' or 'ai_generated'")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    confidence_percent: float = Field(..., description="Confidence as percentage")
    
    # Forensic scores
    ela_score: float = Field(..., ge=0, le=1, description="ELA anomaly score")
    srm_score: float = Field(..., ge=0, le=1, description="SRM noise score")
    dct_score: float = Field(..., ge=0, le=1, description="DCT spectral score")
    composite_forensic_score: float = Field(..., description="Combined forensic score")
    forensic_verdict: str = Field(..., description="Human-readable forensic verdict")
    risk_level: str = Field(..., description="'low', 'medium', or 'high'")
    
    # Visualization images (base64 encoded PNG)
    original_image: str = Field(..., description="Original image as base64")
    ela_image: str = Field(..., description="ELA visualization as base64")
    srm_image: str = Field(..., description="SRM visualization as base64")
    dct_image: str = Field(..., description="DCT visualization as base64")
    gradcam_image: str = Field(..., description="Grad-CAM heatmap overlay as base64")
    
    # Metadata
    model_version: str = Field(default="1.0.0")
    analysis_time_ms: float = Field(..., description="Total analysis time in ms")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    version: str = Field(default="1.0.0")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
