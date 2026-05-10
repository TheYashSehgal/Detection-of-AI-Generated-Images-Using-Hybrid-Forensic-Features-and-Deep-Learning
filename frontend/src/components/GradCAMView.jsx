import React from 'react';

export default function GradCAMView({ originalImage, gradcamImage }) {
  return (
    <div className="gradcam-card glass-card">
      <h3>🧠 Grad-CAM Explainability</h3>
      <div className="gradcam-images">
        <div className="gradcam-img-wrapper">
          <img src={`data:image/png;base64,${originalImage}`} alt="Original" />
          <span className="img-label">Original</span>
        </div>
        <div className="gradcam-img-wrapper">
          <img src={`data:image/png;base64,${gradcamImage}`} alt="Grad-CAM Heatmap" />
          <span className="img-label">Grad-CAM</span>
        </div>
      </div>
      <p className="gradcam-description">
        <strong>Gradient-weighted Class Activation Mapping (Grad-CAM)</strong> highlights the regions
        that most influenced the model's decision. Warm colors (red/yellow) indicate areas the model
        focused on when making its prediction. This provides visual evidence of <em>why</em> the
        model classified the image as real or AI-generated.
      </p>
    </div>
  );
}
