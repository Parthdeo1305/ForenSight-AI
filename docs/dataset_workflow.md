# Dataset Workflow — Anti-Gravity Deepfake Detection

## Dataset Sources & Rationale

### 1. FaceForensics++ (35% of training data)
- **Source**: [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- **Size**: ~1.7M faces from 1,000 videos
- **Manipulation types**: Deepfakes (DF), Face2Face (F2F), FaceShifter (FS), NeuralTextures (NT)
- **Why**: Industry gold standard; covers multiple manipulation techniques
- **Download**: Requires agreement to terms and visiting the official GitHub

### 2. DeepFake Detection Challenge (DFDC) (35% of training data)
- **Source**: [kaggle.com/c/deepfake-detection-challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
- **Size**: ~100,000 video clips (10M+ frames)
- **Why**: Largest public deepfake dataset; most diverse in lighting, ethnicity, compression
- **Download**: Kaggle CLI: `kaggle competitions download -c deepfake-detection-challenge`

### 3. Celeb-DF v2 (20% of training data)
- **Source**: [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- **Size**: ~590K frames from 590 videos, 59 celebrities
- **Why**: Very high-quality synthesis; important for testing against near-perfect fakes
- **Download**: Contact authors via GitHub for access

### 4. ForgeryNet (10% of training data)
- **Source**: [github.com/yinanhe/forgerynet](https://github.com/yinanhe/forgerynet)
- **Size**: ~1.4M faces with 7 manipulation types
- **Why**: Diverse forgery types including partial manipulation that others lack
- **Download**: Available at the official GitHub repository

---

## Preprocessing Pipeline

```
Raw Video
    │
    ▼ Step 1: Frame Extraction
    │   OpenCV reads video at 1 FPS (configurable)
    │   Uniform sampling for temporal clips (16 frames)
    │
    ▼ Step 2: Face Detection
    │   Primary: MTCNN (facenet-pytorch)
    │     • Detects face bounding box
    │     • Extracts 5 facial landmarks (eyes, nose, mouth corners)
    │   Fallback: OpenCV Haar cascade
    │
    ▼ Step 3: Face Alignment
    │   Compute eye center line angle
    │   Affine rotation to horizontalize eyes
    │   Ensures consistent face orientation
    │
    ▼ Step 4: Crop & Resize
    │   224×224 pixels
    │   20% margin around face crop
    │
    ▼ Step 5: Save as PNG
    │   datasets/processed/{source}/{real|fake}/{video_name}/frame_NNNN.png
    │
    ▼ Step 6: Generate CSV Manifest
        path,label,source,video,frame
        /path/to/face.png,0,faceforensics,video001,3
        /path/to/fake.png,1,dfdc,dfdc_video123,7
```

---

## Train/Val/Test Splits

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model optimization |
| Val | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final unbiased evaluation |

Splits are **stratified by label** (equal REAL/FAKE ratio in each split).

---

## Data Augmentation

| Augmentation | Probability | Purpose |
|-------------|-------------|---------|
| HorizontalFlip | 0.5 | Mirror-image generalization |
| RandomBrightnessContrast | 0.3 | Lighting variation robustness |
| GaussianBlur | 0.2 | Video encoding quality variation |
| JPEG Compression | 0.3 | Simulate post-processing artifacts |
| GaussNoise | 0.2 | Camera noise robustness |
| CoarseDropout | 0.2 | Force global context reasoning |
| Rotation ±10° | 0.3 | Non-frontal face handling |
| HueSaturationValue | 0.2 | Color space variation |

---

## Class Balance Handling

- **WeightedRandomSampler**: Each batch maintains ~50% REAL / 50% FAKE
- **Label Smoothing (0.1)**: Prevents model from being overconfident on noisy labels
- **Maximum oversample ratio**: 2.0x — prevents data-starved classes from being over-represented
