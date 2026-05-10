# 🔍 Detection of AI-Generated Images Using Hybrid Forensic Features and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev)

> **Major Project Report** — B.Tech in Computer Science & Engineering (2025-26)
>
> Department of Computer Science & Engineering
> Faculty of Engineering and Technology
> **Gurukula Kangri (Deemed to be University), Haridwar, Uttarakhand**

### 👥 Team Members

| Name | Roll Number |
|------|-------------|
| **Aditya Dhiman** | 226310024 |
| **Aditya Bhatia** | 226320009 |
| **Aditya Sharma** | 226320010 |
| **Yash Sehgal** | 226301239 |

### 🎓 Project Guide
**Mr. Kuldeep Giri** — Assistant Professor, Department of CSE

---

## 📖 Abstract

A full-stack web application that detects AI-generated images using a powerful **Smart Hybrid Fusion** approach. The system combines the semantic intelligence of a state-of-the-art Vision Transformer (**OpenAI CLIP ViT-B/32**) via zero-shot classification with classical digital image forensics — **Error Level Analysis (ELA)**, **SRM Noise Filters**, and **DCT Spectral Analysis**. The model's decisions are explainable through **Attention Rollout Heatmaps**, providing visual evidence of which image regions the transformer focused on to make its classification.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────┐
│                 React Frontend                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │  Upload   │ │ Results  │ │  Forensic Viz    ││
│  │  Zone     │ │ Dashboard│ │  (ELA/SRM/DCT)   ││
│  └──────────┘ └──────────┘ └──────────────────┘│
└──────────────────────┬──────────────────────────┘
                       │ REST API
┌──────────────────────┴──────────────────────────┐
│                FastAPI Backend                    │
│  ┌───────────────────────────────────────────┐  │
│  │         Hybrid Inference Engine           │  │
│  │                                           │  │
│  │  ┌─────────────────┐ ┌─────────────────┐  │  │
│  │  │   OpenAI CLIP   │ │ Classical Image │  │  │
│  │  │   (ViT-B/32)    │ │   Forensics     │  │  │
│  │  │   Zero-Shot     │ │ (ELA, SRM, DCT) │  │  │
│  │  └────────┬────────┘ └────────┬────────┘  │  │
│  │     75%   │                   │  25%      │  │
│  │           ▼                   ▼           │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │      Weighted Probability Fusion    │  │  │
│  │  └──────────────────┬──────────────────┘  │  │
│  │                     ▼                     │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │ Explainability: Attention Rollout   │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Zero-Shot Deep Learning** | Uses OpenAI CLIP to understand images semantically without needing task-specific retraining. |
| **3 Classical Forensic Techniques** | Cross-validates predictions using Error Level Analysis, SRM Noise Filters, and DCT Spectral Analysis. |
| **Attention Rollout Heatmaps** | Visual overlays showing exactly where the Vision Transformer focused its attention. |
| **Smart Hybrid Fusion** | Calculates final probability using a 75/25 weighted fusion of semantic AI and statistical image analysis. |
| **Premium Dark UI** | Modern glassmorphism design with responsive components and smooth animations. |
| **Cloud Ready** | Containerized backend designed for Hugging Face Spaces (16GB RAM) and static frontend for Vercel. |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Git

### Local Development

**1. Backend:**
```bash
cd backend
pip install -r requirements.txt

# Run the server (First boot downloads the ~600MB CLIP model)
uvicorn app.main:app --reload --port 8000
```
*Tip: Set the `HF_HOME` environment variable to change where the CLIP model is cached.*

**2. Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## 📁 Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── routes/              # API endpoints
│   │   ├── forensics/           # ELA, SRM, DCT classical algorithms
│   │   └── ml/
│   │       ├── inference.py     # Hybrid Engine (CLIP + Forensics)
│   │       └── gradcam.py       # ViT Attention Rollout
│   ├── requirements.txt
│   └── Dockerfile               # HF Spaces Deployment Config
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React app
│   │   ├── index.css            # Design system
│   │   └── components/          # UI components
│   └── package.json
└── README.md
```

## 🔬 Technical Details

### Deep Learning Pipeline (OpenAI CLIP)
- **Architecture:** Vision Transformer (ViT-B/32).
- **Strategy:** Zero-Shot Classification computing cosine similarity between image embeddings and specifically engineered text prompts ("a real photograph..." vs "an AI-generated image...").
- **Output:** Calibrated probability tensor.

### Forensic Feature Extraction
- **ELA (Error Level Analysis):** Re-compresses at JPEG quality 90 and computes pixel-wise difference. AI images often exhibit uniform compression.
- **SRM (Spatial Rich Model):** Applies high-pass filter kernels to extract noise residuals. AI images lack natural sensor noise fingerprints.
- **DCT (Discrete Cosine Transform):** Block-wise (8×8) spectral analysis. AI images show anomalous high-frequency distributions.

### Explainability
- **Attention Rollout:** Extracts self-attention matrices from the transformer layers.
- **Heatmap:** Aggregates attention scores across heads and layers to highlight regions of the image that contributed most strongly to the classification verdict.

## 🧪 API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check — verifies API and model status |
| `/api/analyze` | POST | Upload image → returns prediction, confidence, forensic scores, and attention map |

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | PyTorch 2.x, Hugging Face Transformers, OpenAI CLIP |
| **Forensics** | NumPy, SciPy, OpenCV, Pillow |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | React 18, Vite |
| **Explainability** | Transformer Attention Rollout |
| **Deployment Target** | Hugging Face Spaces (Backend), Vercel (Frontend) |

---

### 📜 License
Submitted for the partial fulfilment of the award of **Bachelor of Technology in Computer Science & Engineering** at Gurukula Kangri (Deemed to be University), Haridwar, Uttarakhand — 2025-26.
