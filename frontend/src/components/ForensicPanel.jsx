import React, { useState } from 'react';

const TABS = [
  {
    key: 'ela',
    label: 'ELA',
    title: 'Error Level Analysis',
    description:
      'ELA re-compresses the image and measures pixel-wise differences. Regions with inconsistent compression levels light up — a hallmark of AI-generated content that lacks natural JPEG compression history.',
  },
  {
    key: 'srm',
    label: 'SRM',
    title: 'SRM Noise Analysis',
    description:
      'Spatial Rich Model filters extract high-frequency noise residuals. Camera-captured images have characteristic sensor noise patterns, while AI-generated images exhibit unnaturally smooth or periodic noise.',
  },
  {
    key: 'dct',
    label: 'DCT',
    title: 'DCT Spectral Analysis',
    description:
      'Discrete Cosine Transform reveals frequency domain characteristics. AI-generated images often show anomalous spectral energy distributions compared to natural photographs.',
  },
];

function ScoreBar({ label, score }) {
  const percent = Math.round(score * 100);
  const level = score > 0.6 ? 'high' : score > 0.35 ? 'medium' : 'low';
  return (
    <div className="score-bar-container">
      <div className="score-bar-label">
        <span>{label}</span>
        <span style={{ color: level === 'high' ? '#ff4757' : level === 'medium' ? '#ffa502' : '#00d4ff' }}>
          {percent}%
        </span>
      </div>
      <div className="score-bar">
        <div className={`score-bar-fill ${level}`} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

export default function ForensicPanel({ result }) {
  const [activeTab, setActiveTab] = useState('ela');
  const tab = TABS.find((t) => t.key === activeTab);
  const imageKey = `${activeTab}_image`;
  const scoreKey = `${activeTab}_score`;

  return (
    <div className="forensic-card glass-card">
      <h3>🔬 Forensic Feature Analysis</h3>
      <div className="forensic-tabs">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`forensic-tab ${activeTab === t.key ? 'active' : ''}`}
            onClick={() => setActiveTab(t.key)}
            id={`tab-${t.key}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="forensic-content" key={activeTab}>
        <img
          src={`data:image/png;base64,${result[imageKey]}`}
          alt={tab.title}
          className="forensic-image"
        />
        <p className="forensic-description">
          <strong>{tab.title}:</strong> {tab.description}
        </p>
        <ScoreBar label={`${tab.label} Anomaly Score`} score={result[scoreKey]} />
      </div>
    </div>
  );
}
