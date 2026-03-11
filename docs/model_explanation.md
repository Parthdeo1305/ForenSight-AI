# Model Architecture — Anti-Gravity Deepfake Detection

## Why Three Models?

Each model exploits fundamentally different signals:

| Model | Signal Type | What It Detects | Strength |
|-------|-------------|-----------------|----------|
| EfficientNet-B4 | Local spatial | Blending artifacts, skin anomalies | Texture-level forgery |
| ViT-B/16 | Global attention | Long-range inconsistencies, structural violations | Global context |
| CNN+BiLSTM | Temporal | Blinking, lip sync, frame drift | Video authenticity |

No single model catches everything. The ensemble combines all three for maximum coverage.

---

## Model 1: EfficientNet-B4 (Spatial CNN)

### Architecture

```
Input (1, 3, 224, 224)
    │
    ▼ EfficientNet-B4 backbone (pretrained ImageNet)
    │   - 9 MBConv blocks with compound scaling
    │   - Squeeze-and-Excitation attention per block
    │   - Global Average Pooling → 1792-dim vector
    │
    ▼ Custom Classification Head
    │   Dropout(0.4)
    │   → Linear(1792 → 512) + ReLU + BatchNorm
    │   → Dropout(0.3)
    │   → Linear(512 → 128) + ReLU
    │   → Linear(128 → 1)
    │
    ▼ Sigmoid → Deepfake probability
```

### Why EfficientNet-B4?
- **Compound scaling**: simultaneously scales depth, width, and resolution — captures both fine-grained textures (blending artifacts) and abstract features (skin consistency)
- **~19M parameters**: best accuracy/efficiency trade-off for face images
- **Pretrained on ImageNet-21k**: rich low-level feature representations that transfer directly to forgery detection

### What It Detects
- GAN blending seams at the jaw and forehead boundary
- Unnatural skin texture and lighting gradients
- Frequency-domain artifacts from neural rendering
- Color inconsistencies between face and background

---

## Model 2: Vision Transformer (ViT-B/16)

### Architecture

```
Input (1, 3, 224, 224)
    │
    ▼ Patch Embedding: 196 patches of 16×16 pixels
    │   Each patch → 768-dim token
    │
    ▼ Learnable [CLS] token prepended
    │
    ▼ 12 Transformer Encoder Blocks
    │   Each block:
    │     Multi-Head Self-Attention (12 heads)
    │     → LayerNorm → MLP → LayerNorm
    │     All 197 tokens attend to each other
    │
    ▼ Extract [CLS] token (768-dim)
    │
    ▼ Custom Classification Head
    │   LayerNorm(768)
    │   → Linear(768 → 256) + GELU
    │   → Dropout(0.3)
    │   → Linear(256 → 64) + GELU
    │   → Linear(64 → 1)
    │
    ▼ Sigmoid → Deepfake probability
```

### Why Vision Transformer?
- **Self-attention**: every patch attends to every other patch simultaneously — CNNs process local neighborhoods first, potentially missing long-range violations (e.g., ears inconsistent with face angle)
- **No inductive bias**: doesn't assume spatial locality; learns arbitrary patch relationships
- **Pretrained on ImageNet-21k + ImageNet-1k**: rich semantic understanding of face structure

### What It Detects
- Face orientation inconsistencies (eyes vs. ear position)
- Global lighting direction violations
- Identity consistency across distant face regions
- Structural manipulation patterns too subtle for local convolutions

---

## Model 3: Temporal CNN+BiLSTM

### Architecture

```
Input: Video clip (B, T=16, 3, 224, 224)
    │
    ▼ Frame Feature Extraction (CNN)
    │   ResNet-18 backbone (pretrained)
    │   Remove last FC layer → 512-dim feature/frame
    │   AdaptiveAvgPool2d(1, 1) + Flatten
    │   → (B, T, 512)
    │
    ▼ Bidirectional LSTM (2 layers)
    │   Input: 512-dim per frame
    │   Hidden: 256 per direction → 512 effective
    │   Captures both forward and backward temporal patterns
    │
    ▼ Concatenate final hidden states (forward + backward)
    │   h_forward[-1] + h_backward[-1] → 512-dim
    │
    ▼ Classification Head
    │   LayerNorm(512)
    │   → Linear(512 → 128) + ReLU + Dropout(0.3)
    │   → Linear(128 → 1)
    │
    ▼ Sigmoid → Deepfake probability
```

### Why BiLSTM over GRU?
- LSTM gates (input, forget, output) provide better control over long-term dependencies
- Bidirectionality considers future frames to contextualize past — important for blinking patterns where the full blink sequence (close → open) must be seen
- Orthogonal initialization prevents vanishing/exploding gradients in deep LSTMs

### What It Detects
- **Blinking anomalies**: deepfakes often have irregular or absent blink patterns
- **Lip synchronization**: rendering artifacts cause temporal desynchronization with speech
- **Temporal flickering**: GAN generators sometimes produce inconsistent frames
- **Identity drift**: subtle shifts in appearance over time that no single frame reveals

---

## Ensemble Combination

### Weighting Strategy

| Model | Weight | Justification |
|-------|--------|---------------|
| EfficientNet-B4 | **40%** | Best single-frame performance; direct spatial evidence |
| ViT-B/16 | **30%** | Complements CNN with global context; slightly lower recall on small crops |
| CNN+BiLSTM | **30%** | Essential for video; not applicable to static images |

For **image inference**: Temporal model excluded, weights renormalized to CNN=57%, ViT=43%.

### Confidence Score

```
confidence = min(|probability - 0.5| × 2, 1.0)
```

- Score of 1.0: prediction at extreme (0% or 100% probability) — very confident
- Score of 0.0: prediction exactly at decision boundary (50%) — highly uncertain

### Model Agreement

```
agreement = 1 - std(cnn_score, vit_score, temporal_score)
```

High agreement (>0.9) means all models agree — reliable prediction.
Low agreement (<0.5) suggests ambiguous media — treat with caution.

---

## Grad-CAM Explainability

**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights which parts of the face influenced the prediction:

1. Register forward hook on the last convolutional block of EfficientNet-B4
2. Forward pass → save feature maps from target layer
3. Backward pass → save gradients w.r.t. those feature maps
4. Compute importance weights: `w_c = GlobalAvgPool(∂score/∂A_k^c)`
5. Weighted sum: `L_Grad-CAM = ReLU(Σ w_c · A^c)`
6. Upsample + normalize + overlay on original image with JET colormap

**Interpretation**: 🔴 Red = high manipulation probability region · 🔵 Blue = neutral
