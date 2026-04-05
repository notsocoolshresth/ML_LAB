# Landslide Detection using Multi-Scale Attention U-Net

## Assignment 2: Deep Learning for Semantic Segmentation
### Landslide4Sense (L4S) Benchmark

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (auto-downloads dataset)
python train.py --download --data_dir ./data/Landslide4sense --epochs 100 --batch_size 16

# 3. Or on Google Colab (recommended):
#    Upload this folder, then run:
#    !python train.py --download --epochs 100 --batch_size 16
```

---

## Part A: Model Design

### Architecture: UNet++ with ResNet34 Encoder, scSE Attention, and Spectral Index Features

**Input Processing Pipeline:**
```
14-band Sentinel-2/ALOS input (128×128×14)
        │
        ├──► Spectral Index Module ──► NDVI, NDWI, NDBI, BSI (4 channels)
        │
        ▼
18-channel fused input (128×128×18)
        │
        ▼
Input Projection Network (Conv 7×7 → Conv 3×3 → Conv 1×1)
        │   Maps 18ch → 64ch → 32ch → 3ch
        │   (Learns optimal band combination for pretrained encoder)
        ▼
┌───────────────────────────────────────────┐
│         UNet++ with ResNet34 Encoder       │
│                                           │
│  Encoder (ResNet34, ImageNet pretrained):  │
│    Stage 1: 64ch,  64×64                  │
│    Stage 2: 64ch,  32×32                  │
│    Stage 3: 128ch, 16×16                  │
│    Stage 4: 256ch, 8×8                    │
│    Stage 5: 512ch, 4×4                    │
│                                           │
│  Decoder (UNet++ dense skip connections): │
│    + scSE attention at each level         │
│    Reconstructs to 128×128               │
│                                           │
│  Segmentation Head: Conv 1×1 → 1ch       │
└───────────────────────────────────────────┘
        │
        ▼
Binary Landslide Mask (128×128×1)
```

### Key Design Contributions

1. **Handcrafted Spectral Indices (4 extra channels):**
   - NDVI = (NIR - Red) / (NIR + Red) → vegetation health
   - NDWI = (Green - NIR) / (Green + NIR) → water content
   - NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) → built-up areas
   - BSI = ((SWIR1 + Red) - (NIR + Green)) / ((SWIR1 + Red) + (NIR + Green)) → bare soil
   
   **Justification:** Landslides strip vegetation (low NDVI), expose soil (high BSI),
   and alter water patterns (NDWI changes). These indices encode domain-specific 
   knowledge that raw bands alone don't explicitly capture.

2. **Input Projection Network (18ch → 3ch):**
   Learns an optimal linear combination of all 18 channels to map into the 3-channel
   space expected by the ImageNet-pretrained encoder. This is superior to simply 
   discarding bands or using the first 3 channels.

3. **UNet++ (Nested U-Net) instead of vanilla U-Net:**
   Dense skip connections between encoder and decoder allow the model to capture 
   features at multiple semantic scales simultaneously, critical for detecting 
   landslides of varying sizes.

4. **scSE (Spatial + Channel Squeeze-and-Excitation) Attention:**
   Adaptively recalibrates both channel-wise and spatial feature responses in the 
   decoder, helping the model focus on landslide-relevant patterns.

5. **Combined Tversky + Focal Loss:**
   - Tversky Loss (α=0.3, β=0.7): Penalizes false negatives more than false 
     positives → higher recall for detecting landslide pixels
   - Focal Loss (α=0.75, γ=2.0): Down-weights easy background samples → focuses 
     learning on hard examples near landslide boundaries

6. **Training Strategy:**
   - OneCycleLR scheduler for super-convergence
   - Strong spatial augmentation (flips, rotations, shift-scale-rotate, noise)
   - Gradient clipping (max_norm=1.0)
   - Early stopping with patience=20 on Z-score (0.6×F1 + 0.4×IoU)

### Trainable Parameters
~25.7M parameters (ResNet34 encoder + UNet++ decoder + input projection + scSE attention)

---

## Part B: Performance Metrics

Training targets test-set metrics that exceed the baseline:

| Metric        | Baseline  | Target (Full Marks) |
|--------------|-----------|---------------------|
| F1 Score     | 74.54%    | ≥ 78%              |
| IoU          | 59.45%    | ≥ 62%              |
| Precision    | 74.30%    | > 75%              |
| Recall       | 74.80%    | > 75%              |

After training with the above architecture, expected results:

| Metric        | Expected   |
|--------------|------------|
| Precision    | ~78-82%    |
| Recall       | ~79-84%    |
| F1 Score     | ~79-82%    |
| IoU          | ~65-70%    |

---

## Part C: Efficiency Metric

```
Z = 0.6 × F1% + 0.4 × IoU%
```

With expected F1 ≈ 80% and IoU ≈ 67%:
```
Z = 0.6 × 80.0 + 0.4 × 67.0 = 48.0 + 26.8 = 74.8
```

---

## Project Structure

```
landslide_project/
├── dataset.py          # Data loading, augmentation, spectral indices
├── model.py            # Model architectures (UNet++, custom LandslideSegNet)
├── losses.py           # Dice, Focal, Tversky, Combined losses
├── train.py            # Full training pipeline
├── requirements.txt    # Python dependencies
└── README.md           # This file (documentation)
```

---

## Running on Google Colab (Recommended)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install deps
!pip install segmentation-models-pytorch albumentations h5py huggingface_hub

# Upload project files or clone from your repo
# Then run:
!python train.py --download --data_dir /content/data/Landslide4sense \
    --output_dir /content/drive/MyDrive/landslide_checkpoints \
    --epochs 100 --batch_size 16 --lr 1e-3 --encoder resnet34
```

---

## Viva Preparation (Part D)

Key points to articulate:

1. **Why UNet++ over U-Net?** Dense skip connections capture multi-scale features; 
   landslides vary from small debris flows to large slope failures.

2. **Why pretrained + projection?** ImageNet features capture universal visual patterns
   (edges, textures). The projection layer learns which spectral band combinations 
   are most informative.

3. **Why spectral indices?** They encode known physical relationships between 
   landslide occurrence and surface properties (vegetation removal, soil exposure).

4. **Why Tversky + Focal?** Landslide pixels are rare (~5-10% of the image).
   Tversky's asymmetric weighting prioritizes catching all landslide pixels,
   while Focal concentrates on hard boundary pixels.

5. **Parameter efficiency:** ~25.7M params is reasonable for this task. The ResNet34
   encoder is relatively lightweight vs. ResNet50/101. scSE attention adds minimal
   overhead (~1% extra params) but meaningfully improves spatial focus.
