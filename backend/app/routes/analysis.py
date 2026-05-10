"""Image analysis API endpoint."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import AnalysisResponse
from app.ml.inference import engine
from app.forensics.pipeline import get_forensic_summary
from app.utils.image_utils import load_image_from_bytes, image_to_base64
from app.config import MAX_FILE_SIZE

router = APIRouter(tags=["Analysis"])

@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image for AI generation indicators."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, WebP)")

    # Read and validate size
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")

    try:
        image = load_image_from_bytes(data)
    except Exception:
        raise HTTPException(400, "Could not open image file")

    # Run prediction
    try:
        result = engine.predict(image)
    except RuntimeError as e:
        raise HTTPException(503, f"Model error: {str(e)}")

    # Get forensic summary (use the hybrid AI probability for verdict/risk)
    summary = get_forensic_summary({
        'ela_score': result['ela_score'],
        'srm_score': result['srm_score'],
        'dct_score': result['dct_score'],
        'ai_probability': result.get('ai_probability', 0.5),
    })

    # Encode images to base64
    return AnalysisResponse(
        prediction=result['prediction'],
        confidence=result['confidence'],
        confidence_percent=result['confidence_percent'],
        ela_score=result['ela_score'],
        srm_score=result['srm_score'],
        dct_score=result['dct_score'],
        composite_forensic_score=summary['composite_score'],
        forensic_verdict=summary['verdict'],
        risk_level=summary['risk_level'],
        original_image=image_to_base64(image),
        ela_image=image_to_base64(result['ela_image']),
        srm_image=image_to_base64(result['srm_image']),
        dct_image=image_to_base64(result['dct_image']),
        gradcam_image=result['gradcam_image'],
        analysis_time_ms=result['analysis_time_ms'],
    )
