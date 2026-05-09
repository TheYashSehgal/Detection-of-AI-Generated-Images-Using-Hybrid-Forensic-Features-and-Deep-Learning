# 🔍 Detection of AI-Generated Images Using Hybrid Forensic Features and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
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

A full-stack web application that detects AI-generated images using a hybrid approach combining classical digital image forensics with modern deep learning. The system employs three forensic techniques — **Error Level Analysis (ELA)**, **SRM Noise Filters**, and **DCT Spectral Analysis** — fused with an **EfficientNet-B0** backbone through **Squeeze-and-Excitation** attention blocks. The model's decisions are explained through **Grad-CAM** heatmaps, providing visual evidence of which image regions influenced the classification.

---

## 🏗️ Architecture

```
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
│  │         Forensic Pipeline                  │  │
│  │  ┌─────┐  ┌─────┐  ┌─────┐              │  │
│  │  │ ELA │  │ SRM │  │ DCT │              │  │
│  │  └──┬──┘  └──┬──┘  └──┬──┘              │  │
│  │     └────────┼────────┘                   │  │
│  │              ▼                            │  │
│  │  ┌───────────────────────┐               │  │
│  │  │  Hybrid CNN Model     │               │  │
│  │  │  ┌─────────────────┐  │               │  │
│  │  │  │ EfficientNet-B0 │──┤               │  │
│  │  │  └─────────────────┘  │  SE Fusion    │  │
│  │  │  ┌─────────────────┐  │──► Classifier │  │
│  │  │  │ Forensic CNN    │──┤               │  │
│  │  │  └─────────────────┘  │               │  │
│  │  └───────────────────────┘               │  │
│  │              │                            │  │
│  │              ▼                            │  │
│  │        Grad-CAM Heatmap                   │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Architecture** | Dual-branch CNN combining EfficientNet-B0 (semantic) + forensic CNN (noise patterns) |
| **3 Forensic Techniques** | Error Level Analysis, SRM Noise Filters, DCT Spectral Analysis |
| **Grad-CAM Explainability** | Visual heatmaps showing which regions influenced the model's decision |
| **SE Fusion** | Squeeze-and-Excitation blocks for adaptive feature fusion |
| **Premium Dark UI** | Modern glassmorphism design with smooth animations |
| **Docker Deployment** | One-command deployment with Docker Compose |
| **REST API** | FastAPI with automatic OpenAPI documentation |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (for deployment)

### Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Train the Model:**
```bash
cd backend
python -m app.ml.training --data-dir ./data --generate-data --epochs 30 --device cpu
```

### Docker Deployment
```bash
docker compose up --build
```
Open http://localhost in your browser.

## 📁 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── routes/              # API endpoints
│   │   ├── models/              # CNN model architecture
│   │   ├── forensics/           # ELA, SRM, DCT modules
│   │   ├── ml/                  # Inference, training, Grad-CAM
│   │   └── utils/               # Image utilities
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React app
│   │   ├── index.css            # Design system
│   │   ├── components/          # UI components
│   │   └── api/                 # API client
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── render.yaml
└── README.md
```

## 🔬 Technical Details

### Forensic Feature Extraction
- **ELA**: Re-compresses at JPEG quality 90, computes pixel-wise difference
- **SRM**: Applies 23+ high-pass filter kernels to extract noise residuals
- **DCT**: Block-wise (8×8) Discrete Cosine Transform for spectral analysis

### Model Architecture
- **Branch A**: EfficientNet-B0 (pre-trained on ImageNet) — 1280-dim semantic features
- **Branch B**: 4-layer CNN with BatchNorm — 256-dim forensic features
- **Fusion**: SE block for channel-wise feature recalibration
- **Total Parameters**: ~6M (lightweight, CPU-deployable)

### Explainability
- Grad-CAM on EfficientNet's last convolutional layer
- Generates heatmap overlay highlighting influential image regions

## 🧪 API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive Swagger UI.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check — verify model is loaded |
| `/api/analyze` | POST | Upload image → returns prediction, forensic scores, Grad-CAM |

## 📊 Training

Organize your dataset as:
```
data/
├── real/       # Authentic photographs
└── fake/       # AI-generated images
```

Run:
```bash
python -m app.ml.training --data-dir ./data --epochs 50 --device cuda
```

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.x, EfficientNet-B0 |
| Forensics | NumPy, SciPy, OpenCV, Pillow |
| Backend | FastAPI, Uvicorn |
| Frontend | React 18, Vite |
| Explainability | Grad-CAM |
| Deployment | Docker, Render.com |

---

### 📜 License
Submitted for the partial fulfilment of the award of **Bachelor of Technology in Computer Science & Engineering** at Gurukula Kangri (Deemed to be University), Haridwar, Uttarakhand — 2025-26.
