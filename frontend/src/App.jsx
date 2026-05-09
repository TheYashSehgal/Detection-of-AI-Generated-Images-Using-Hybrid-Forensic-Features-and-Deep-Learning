import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import LoadingSpinner from './components/LoadingSpinner';
import AnalysisResult from './components/AnalysisResult';
import FeatureCards from './components/FeatureCards';
import Footer from './components/Footer';
import { analyzeImage } from './api/client';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await analyzeImage(selectedFile);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }, []);

  return (
    <>
      <Header />

      {!result && !isLoading && (
        <>
          <section className="hero">
            <h2>
              Detect <span className="highlight">AI-Generated</span> Images
            </h2>
            <p>
              Upload any image to analyze it using hybrid forensic features
              (ELA, SRM, DCT) and deep learning with Grad-CAM explainability.
            </p>
          </section>

          <ImageUploader
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            preview={preview}
            onAnalyze={handleAnalyze}
            isLoading={isLoading}
          />

          {error && (
            <div style={{
              maxWidth: 700, margin: '0 auto 24px', padding: '16px 24px',
              background: 'rgba(255,71,87,0.1)', border: '1px solid rgba(255,71,87,0.3)',
              borderRadius: 12, color: '#ff4757', textAlign: 'center', fontSize: '0.9rem',
            }}>
              ⚠️ {error}
            </div>
          )}

          <FeatureCards />
        </>
      )}

      {isLoading && <LoadingSpinner />}

      {result && (
        <AnalysisResult result={result} onReset={handleReset} />
      )}

      <Footer />
    </>
  );
}
