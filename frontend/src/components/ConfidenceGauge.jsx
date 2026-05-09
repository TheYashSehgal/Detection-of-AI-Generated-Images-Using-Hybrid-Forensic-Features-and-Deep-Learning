import React, { useEffect, useState } from 'react';

export default function ConfidenceGauge({ confidence, prediction, analysisTime }) {
  const [animatedValue, setAnimatedValue] = useState(0);
  const radius = 85;
  const circumference = 2 * Math.PI * radius;
  const percent = Math.round(animatedValue * 100);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedValue(confidence), 300);
    return () => clearTimeout(timer);
  }, [confidence]);

  const offset = circumference - animatedValue * circumference;
  const gaugeClass = prediction === 'real' ? 'real' : confidence > 0.7 ? 'ai' : 'uncertain';

  return (
    <div className="gauge-card glass-card">
      <div className="gauge-container">
        <svg className="gauge-svg" viewBox="0 0 200 200">
          <circle className="gauge-bg" cx="100" cy="100" r={radius} />
          <circle
            className={`gauge-fill ${gaugeClass}`}
            cx="100" cy="100" r={radius}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="gauge-center">
          <div className={`gauge-value ${gaugeClass}`}>{percent}%</div>
          <div className="gauge-label">confidence</div>
        </div>
      </div>
      <div className="gauge-title">
        {prediction === 'real' ? '✅ Authentic Image' : '⚠️ AI-Generated'}
      </div>
      <div className="gauge-subtitle">
        {prediction === 'real'
          ? 'This image appears to be a genuine photograph'
          : 'This image shows indicators of AI generation'}
      </div>
      <div className="analysis-time">⏱️ {analysisTime}ms</div>
    </div>
  );
}
