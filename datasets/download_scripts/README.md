# Dataset Download Instructions

## Overview

The Anti-Gravity system uses 4 datasets. Due to licensing restrictions, 
all datasets require agreement to terms before downloading.

---

## 1. FaceForensics++ (35% of Training Data)

**Description**: Videos with 4 manipulation types — Deepfakes, Face2Face, FaceShifter, NeuralTextures

**Download**:
```bash
# Clone the official release script
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# Request download access at: https://github.com/ondyari/FaceForensics#access
# Then run:
python download-FaceForensics.py /path/to/save/faceforensics -d all -c c23 -t videos
```

**Expected Directory Structure**:
```
faceforensics/
├── original_sequences/youtube/c23/videos/    ← REAL
└── manipulated_sequences/
    ├── Deepfakes/c23/videos/
    ├── Face2Face/c23/videos/
    ├── FaceShifter/c23/videos/
    └── NeuralTextures/c23/videos/            ← FAKE
```

---

## 2. DeepFake Detection Challenge (DFDC) (35% of Training Data)

**Description**: Facebook AI deepfake competition dataset — largest and most diverse

**Download**:
```bash
# Install Kaggle CLI
pip install kaggle

# Download (requires Kaggle account + competition acceptance)
kaggle competitions download -c deepfake-detection-challenge -p /path/to/dfdc

# Extract
cd /path/to/dfdc
for f in *.zip; do unzip "$f"; done
```

**Expected Directory Structure**:
```
dfdc/
├── dfdc_train_part_0/
│   ├── *.mp4           ← Videos (mixed real and fake)
│   └── metadata.json   ← Labels per video
└── ...
```

**Note**: There is no separate real/fake directory. Use `metadata.json` (`"label": "REAL"` or `"FAKE"`) to route files.

---

## 3. Celeb-DF v2 (20% of Training Data)

**Description**: High-quality deepfakes of 59 celebrities

**Download**:
```bash
# Request access at: https://github.com/yuezunli/celeb-deepfakeforensics
# Complete Google Form → receive download link

wget -O celebdf.zip "<PROVIDED_URL>"
unzip celebdf.zip
```

**Expected Directory Structure**:
```
celebdf/
├── Celeb-real/         ← REAL
├── Celeb-synthesis/    ← FAKE (deep fakes of Celeb-real)
├── YouTube-real/       ← Additional real videos
└── List_of_testing_videos.txt
```

---

## 4. ForgeryNet (10% of Training Data)

**Description**: 7 manipulation types including partial face forgery

**Download**:
```bash
# Visit: https://github.com/yinanhe/forgerynet
# Follow download instructions; typically via Baidu Pan or Google Drive link

# After download:
unzip ForgeryNet.zip
```

**Expected Directory Structure**:
```
forgerynet/
├── real/               ← REAL
└── fake/               ← FAKE
```

---

## Running Preprocessing After Download

```bash
python datasets/preprocessing/preprocess.py \
    --dataset_root /path/to/all/datasets \
    --output_root datasets/processed \
    --fps 1.0 \
    --max_frames 30 \
    --num_workers 8 \
    --datasets faceforensics dfdc celebdf forgerynet

# Output: datasets/manifests/train.csv, val.csv, test.csv
```
