import React, { useRef, useState } from 'react';

export default function ImageUploader({ onFileSelect, selectedFile, preview, onAnalyze, isLoading }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      onFileSelect(file);
    }
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  };

  return (
    <section className="uploader-section">
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !selectedFile && inputRef.current?.click()}
        id="upload-zone"
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          id="file-input"
        />

        {!selectedFile ? (
          <>
            <div className="upload-icon">📤</div>
            <h3>Drop your image here</h3>
            <p>or click to browse • Supports JPG, PNG, WebP (max 10MB)</p>
            <button className="browse-btn" onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}>
              Browse Files
            </button>
          </>
        ) : (
          <div className="preview-container" onClick={(e) => e.stopPropagation()}>
            <img src={preview} alt="Preview" className="preview-image" />
            <div className="preview-info">
              <span>📄 {selectedFile.name}</span>
              <span>📐 {formatSize(selectedFile.size)}</span>
            </div>
          </div>
        )}
      </div>

      {selectedFile && (
        <button
          className={`analyze-btn ${isLoading ? 'loading' : ''}`}
          onClick={onAnalyze}
          disabled={isLoading}
          id="analyze-btn"
        >
          {isLoading ? '⏳ Analyzing...' : '🔬 Analyze Image'}
        </button>
      )}
    </section>
  );
}
