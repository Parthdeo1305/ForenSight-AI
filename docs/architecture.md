# Architecture — Anti-Gravity Deepfake Detection System

## System Architecture Diagram

```mermaid
graph TD
    A[User Uploads Media] --> B{Media Type?}
    B -- Image --> C[Save to uploads/]
    B -- Video --> C
    C --> D[FastAPI /detect endpoint]
    D --> E[Inference Engine]

    E --> F[Face Detection\nMTCNN + Haar Fallback]
    F --> G{Detection Mode}

    G -- Image --> H[EfficientNet-B4\nSpatial CNN]
    G -- Image --> I[ViT-B/16\nVision Transformer]
    G -- Video --> H
    G -- Video --> I
    G -- Video --> J[ResNet18 + BiLSTM\nTemporal Model]

    H -- Score 0.40 --> K[Weighted Ensemble]
    I -- Score 0.30 --> K
    J -- Score 0.30 --> K

    K --> L[Final Probability]
    K --> M[Grad-CAM Heatmap\nEfficientNet Target Layer]

    L --> N[Structured JSON Response]
    M --> N

    N --> O[React Frontend]
    O --> P[Probability Gauge]
    O --> Q[Per-Model Scores]
    O --> R[Heatmap Overlay]
    O --> S[Confidence Level]
```

---

## Data Pipeline

```mermaid
graph LR
    A[Raw Datasets] --> B[download_scripts/]
    B --> C[Video Files]
    C --> D[Frame Extraction\nOpenCV @ 1 FPS]
    D --> E[Face Detection\nMTCNN]
    E --> F[Face Alignment\n5-Point Landmarks]
    F --> G[Resize 224x224]
    G --> H[PNG Face Images]
    H --> I[CSV Manifest\ntrain/val/test]
    I --> J[DeepfakeDataset\nPyTorch]
    J --> K[Augmentation\nAlbumentations]
    K --> L[Training Loop]
```

---

## Model Training Pipeline

```mermaid
graph TD
    A[config.yaml] --> B[Build Model\ncnn/vit/temporal]
    B --> C[Load Manifests]
    C --> D[WeightedRandomSampler\nClass Balance]
    D --> E[Training Loop]

    E --> F[Forward Pass\nAMP Mixed Precision]
    F --> G[BCEWithLogitsLoss\nLabel Smoothing]
    G --> H[Backward Pass]
    H --> I[Grad Clip + AdamW Step]
    I --> J[CosineAnnealingLR\n+ Linear Warmup]
    J --> K{Val Loss Improved?}
    K -- Yes --> L[Save Best Checkpoint\nweights/*.pth]
    K -- No --> M[EarlyStopping Counter +1]
    M --> N{Patience Exceeded?}
    N -- No --> E
    N -- Yes --> O[Stop Training]
    L --> E

    O --> P[TensorBoard Logs]
    O --> Q[Per-Epoch Metrics]
```

---

## Component Responsibilities

| Component | File | Responsibility |
|-----------|------|---------------|
| CNN Model | `models/cnn_model.py` | Spatial artifact detection via EfficientNet-B4 |
| ViT Model | `models/vit_model.py` | Global inconsistency detection via attention |
| Temporal Model | `models/temporal_model.py` | Time-domain anomaly detection via BiLSTM |
| Ensemble | `models/ensemble_model.py` | Weighted combination of all three models |
| Face Detector | `utils/face_detection.py` | MTCNN face detection + alignment |
| Augmentation | `utils/augmentation.py` | Albumentations training / val transforms |
| Grad-CAM | `utils/gradcam.py` | Explainability heatmap generation |
| Video Utils | `utils/video_utils.py` | Frame extraction from video files |
| Preprocessor | `datasets/preprocessing/preprocess.py` | Raw video → processed face images |
| Training | `training/train.py` | Full training loop with GPU, AMP, early stopping |
| Metrics | `evaluation/metrics.py` | ROC-AUC, F1, ablation study |
| API | `api/app.py` | FastAPI endpoints |
| Inference Engine | `api/inference.py` | Production inference with face detection + heatmap |
| Frontend | `frontend/react-ui/` | React + Vite + TailwindCSS UI |
