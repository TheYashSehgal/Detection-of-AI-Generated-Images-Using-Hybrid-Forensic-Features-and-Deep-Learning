"""
Forensic Feature Pipeline

Orchestrates all forensic analysis methods (ELA, SRM, DCT) into a single
preprocessing pipeline. Generates both model input tensors and visualizations
for the frontend display.
"""
import numpy as np
from PIL import Image
from app.forensics.ela import compute_ela, compute_ela_tensor
from app.forensics.srm import apply_srm_filters, compute_srm_tensor
from app.forensics.dct import compute_dct_features, compute_dct_tensor
from app.config import IMAGE_SIZE


def extract_forensic_features(image: Image.Image) -> dict:
    """
    Run the complete forensic analysis pipeline on an image.
    
    Args:
        image: PIL Image in RGB mode
    
    Returns:
        dict with keys:
            - 'ela_image': PIL Image of ELA visualization
            - 'ela_score': float ELA anomaly score
            - 'srm_image': PIL Image of SRM noise visualization
            - 'srm_score': float SRM noise score
            - 'dct_image': PIL Image of DCT spectral visualization
            - 'dct_score': float DCT anomaly score
            - 'forensic_tensor': numpy array (9, 224, 224) combined forensic features
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Run each forensic analysis
    ela_image, ela_score = compute_ela(image)
    srm_image, srm_score = apply_srm_filters(image)
    dct_image, dct_score = compute_dct_features(image)
    
    # Compute model input tensors
    ela_tensor = compute_ela_tensor(image, target_size=IMAGE_SIZE)   # (3, 224, 224)
    srm_tensor = compute_srm_tensor(image, target_size=IMAGE_SIZE)   # (3, 224, 224)
    dct_tensor = compute_dct_tensor(image, target_size=IMAGE_SIZE)   # (3, 224, 224)
    
    # Combine into a single multi-channel forensic tensor
    forensic_tensor = np.concatenate([ela_tensor, srm_tensor, dct_tensor], axis=0)  # (9, 224, 224)
    
    return {
        'ela_image': ela_image,
        'ela_score': ela_score,
        'srm_image': srm_image,
        'srm_score': srm_score,
        'dct_image': dct_image,
        'dct_score': dct_score,
        'forensic_tensor': forensic_tensor,
    }


def get_forensic_summary(scores: dict) -> dict:
    """
    Generate a human-readable summary of forensic analysis results.
    
    Args:
        scores: dict with ela_score, srm_score, dct_score
    
    Returns:
        dict with summary information
    """
    ela_score = scores.get('ela_score', 0)
    srm_score = scores.get('srm_score', 0)
    dct_score = scores.get('dct_score', 0)
    
    # Weighted average of forensic scores
    composite_score = 0.4 * ela_score + 0.35 * srm_score + 0.25 * dct_score
    
    # Interpretation
    if composite_score > 0.6:
        forensic_verdict = "High likelihood of AI generation"
        risk_level = "high"
    elif composite_score > 0.4:
        forensic_verdict = "Moderate indicators of AI generation"
        risk_level = "medium"
    else:
        forensic_verdict = "Low indicators of AI generation"
        risk_level = "low"
    
    return {
        'composite_score': composite_score,
        'verdict': forensic_verdict,
        'risk_level': risk_level,
        'details': {
            'ela': {
                'score': ela_score,
                'description': 'Error Level Analysis detects compression inconsistencies. '
                              'AI-generated images often show uniform compression levels.',
            },
            'srm': {
                'score': srm_score,
                'description': 'SRM noise analysis reveals statistical fingerprints in image noise. '
                              'AI-generated images have distinct noise patterns.',
            },
            'dct': {
                'score': dct_score,
                'description': 'DCT spectral analysis examines frequency domain characteristics. '
                              'AI-generated images show anomalous spectral distributions.',
            },
        }
    }
