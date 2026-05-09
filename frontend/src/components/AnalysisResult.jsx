import React from 'react';
import ConfidenceGauge from './ConfidenceGauge';
import ForensicPanel from './ForensicPanel';
import GradCAMView from './GradCAMView';

export default function AnalysisResult({ result, onReset }) {
  return (
    <section className="results-section">
      <div className="results-header">
        <h2>Analysis Complete</h2>
      </div>

      {/* Verdict Banner */}
      <div className={`verdict-banner ${result.prediction}`}>
        <div className="verdict-icon">
          {result.prediction === 'real' ? '✅' : '🚨'}
        </div>
        <div className="verdict-text">
          <h3>
            {result.prediction === 'real'
              ? 'Authentic Image Detected'
              : 'AI-Generated Image Detected'}
          </h3>
          <p>{result.forensic_verdict} • Confidence: {result.confidence_percent}%</p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="results-grid">
        <ConfidenceGauge
          confidence={result.confidence}
          prediction={result.prediction}
          analysisTime={result.analysis_time_ms}
        />
        <ForensicPanel result={result} />
      </div>

      {/* Grad-CAM */}
      <div className="results-grid">
        <div className="results-grid-full" style={{ gridColumn: '1 / -1' }}>
          <GradCAMView
            originalImage={result.original_image}
            gradcamImage={result.gradcam_image}
          />
        </div>
      </div>

      {/* New Analysis Button */}
      <div style={{ textAlign: 'center' }}>
        <button className="new-analysis-btn" onClick={onReset} id="new-analysis-btn">
          🔄 Analyze Another Image
        </button>
      </div>
    </section>
  );
}
