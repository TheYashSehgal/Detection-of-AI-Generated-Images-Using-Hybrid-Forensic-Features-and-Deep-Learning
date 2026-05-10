import React from 'react';

export default function Footer() {
  return (
    <footer className="footer">
      <p style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 4 }}>
        Detection of AI-Generated Images Using Hybrid Forensic Features and Deep Learning
      </p>
      <p>B.Tech CSE Major Project • GK(DU), Haridwar</p>
      <p style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
        Yash Sehgal • Aditya Bhatia • Aditya Dhiman • Aditya Sharma
      </p>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
        Under the guidance of Mr. Kuldeep Giri, Assistant Professor, Dept. of CSE
      </p>
      <div className="tech-stack">
        <span className="tech-tag">PyTorch</span>
        <span className="tech-tag">EfficientNet-B0</span>
        <span className="tech-tag">FastAPI</span>
        <span className="tech-tag">React</span>
        <span className="tech-tag">Grad-CAM</span>
        <span className="tech-tag">SRM Filters</span>
      </div>
    </footer>
  );
}
