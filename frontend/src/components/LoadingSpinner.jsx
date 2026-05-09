import React from 'react';

export default function LoadingSpinner() {
  const steps = [
    { text: 'Extracting Error Level Analysis (ELA)', delay: 0 },
    { text: 'Computing SRM Noise Residuals', delay: 1 },
    { text: 'Running DCT Spectral Analysis', delay: 2 },
    { text: 'Processing through Hybrid CNN', delay: 3 },
    { text: 'Generating Grad-CAM Explanation', delay: 4 },
  ];

  const [activeStep, setActiveStep] = React.useState(0);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loading-overlay">
      <div className="spinner-ring" />
      <div className="loading-text">Analyzing Image</div>
      <div className="loading-sub">Running hybrid forensic analysis pipeline...</div>
      <div className="loading-steps">
        {steps.map((step, i) => (
          <div
            key={i}
            className={`loading-step ${i < activeStep ? 'done' : i === activeStep ? 'active' : ''}`}
          >
            <span>{i < activeStep ? '✅' : i === activeStep ? '⏳' : '⬜'}</span>
            <span>{step.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
