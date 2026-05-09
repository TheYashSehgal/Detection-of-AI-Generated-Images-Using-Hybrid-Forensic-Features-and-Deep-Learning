import React from 'react';

const FEATURES = [
  {
    icon: '📊',
    title: 'Error Level Analysis (ELA)',
    description:
      'Detects compression inconsistencies by re-saving images at known JPEG quality levels and analyzing pixel-wise differences. AI-generated images lack natural compression history.',
  },
  {
    icon: '🔊',
    title: 'SRM Noise Forensics',
    description:
      'Applies 30+ high-pass Spatial Rich Model filters to extract noise residuals. Camera sensors leave unique noise fingerprints that AI generators cannot replicate.',
  },
  {
    icon: '📡',
    title: 'DCT Spectral Analysis',
    description:
      'Examines frequency domain characteristics using the Discrete Cosine Transform. AI-generated images show anomalous spectral energy patterns invisible to the human eye.',
  },
];

export default function FeatureCards() {
  return (
    <section className="features-section">
      <h2>Forensic Detection Techniques</h2>
      <div className="features-grid">
        {FEATURES.map((feature, i) => (
          <div key={i} className="feature-card glass-card">
            <div className="feature-icon">{feature.icon}</div>
            <h3>{feature.title}</h3>
            <p>{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
