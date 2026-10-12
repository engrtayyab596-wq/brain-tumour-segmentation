# Brain Tumour Segmentation — U-Net on LGG MRI

> **Project 11 · ML Engineering Portfolio**
> Pixel-wise segmentation of low-grade glioma (LGG) brain MRI slices using a
> full-resolution U-Net trained with Dice loss.
> **Best val Dice 0.8586 · Best val IoU 0.7645** (epoch 45 / 50, Kaggle T4 GPU)

---

## Table of Contents
1. [Dataset](#dataset)
2. [Architecture](#architecture)
3. [Training](#training)
4. [Results](#results)
5. [Project structure](#project-structure)
6. [Quick start](#quick-start)
7. [API reference](#api-reference)
8. [Docker](#docker)
9. [Tests](#tests)

---

## Dataset

| Item | Value |
|------|-------|
| Source | [mateuszbuda/LGG-MRI-Segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) |
| Kaggle path | `/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m` |
| Total pairs | **3 929** image / mask `.tif` pairs |
| Patients | 110 (TCGA LGG cohort) |
| Image size | variable → resized to **256 × 256** |
| Train split | **3 143** (80 %) |
| Val split | **786** (20 %) |
| Class imbalance | ~33 % of slices have any tumour; mean tumour area **3.30 %** |

Each `.tif` file is a FLAIR/T1/T1ce/T2 MRI slice; the matching `_mask.tif` is
a binary tumour annotation (0 / 255).

---

## Architecture

```
U-Net  (in_channels=3, out_channels=1)
  features = [64, 128, 256, 512]
  Total parameters: ~31 M
  Input  : (B, 3, 256, 256)   RGB, ImageNet-normalised
  Output : (B, 1, 256, 256)   raw logit  →  sigmoid  →  threshold 0.5
```

| Layer | Channels | Spatial |
|-------|----------|---------|
| Encoder 1 | 3 → 64 | 256 |
| Encoder 2 | 64 → 128 | 128 |
| Encoder 3 | 128 → 256 | 64 |
| Encoder 4 | 256 → 512 | 32 |
| Bottleneck | 512 → 1024 | 16 |
| Decoder 4 | 1024+512 → 512 | 32 |
| Decoder 3 | 512+256 → 256 | 64 |
| Decoder 2 | 256+128 → 128 | 128 |
| Decoder 1 | 128+64 → 64 | 256 |
| Output conv | 64 → 1 | 256 |

Skip connections concatenate the corresponding encoder feature maps at each
decoder stage (standard U-Net).

---

## Training

```
Loss         : Dice loss  (handles 3.3 % positive-pixel imbalance)
Optimiser    : Adam  lr=1e-4, weight_decay=1e-5
Scheduler    : ReduceLROnPlateau  patience=5, factor=0.5, mode=max
Epochs       : 50
Batch size   : 16  (fits on Kaggle T4 16 GB with ~8-10 GB VRAM)
```

### Augmentation (train only)

```python
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
A.Rotate(limit=15, p=0.5)
A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.3)
A.RandomBrightnessContrast(p=0.3)
A.GaussianBlur(blur_limit=3, p=0.2)
A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
ToTensorV2()
```

Val transform: Resize(256,256) → Normalize → ToTensorV2 (no augmentation).

To run training on Kaggle (T4 GPU):

```python
from src.train import train

history = train(
    data_path   = '/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m',
    output_dir  = 'models',
    num_epochs  = 50,
    batch_size  = 16,
    lr          = 1e-4,
    img_size    = 256,
    val_split   = 0.2,
    device_str  = 'auto',
)
```

---

## Results

| Epoch | Val Dice | Val IoU |
|------:|--------:|--------:|
| 10 | ~0.780 | ~0.645 |
| 20 | ~0.820 | ~0.700 |
| 30 | ~0.840 | ~0.730 |
| 40 | ~0.852 | ~0.758 |
| **45** | **0.8586** | **0.7645** |
| 50 | 0.857 | 0.763 |

The model checkpoint saved at epoch 45 (`models/best_model.pth`) is the one
used for inference. Train / val Dice gap ≈ 0 throughout — the augmentation
pipeline prevented overfitting.

---

## Project structure

```
brain-tumour-segmentation/
├── src/
│   ├── dataset.py        # BrainMRIDataset, augmentation, DataLoader builders
│   ├── model.py          # UNet class (DoubleConv, Down, Up, OutConv)
│   ├── train.py          # train() function, checkpoint saving, history dict
│   ├── evaluate.py       # dice_loss, dice_coefficient, iou_score
│   └── visualize.py      # visualize_predictions, plot_training_curves, overlay_single
├── api/
│   └── main.py           # FastAPI app (/, /info, /predict)
├── tests/
│   └── test_pipeline.py  # pytest suite (model, metrics, dataset, inference, API)
├── models/               # best_model.pth goes here (not committed)
├── outputs/              # visualisation PNGs written here
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# Clone and install
git clone https://github.com/<you>/brain-tumour-segmentation.git
cd brain-tumour-segmentation
pip install -r requirements.txt

# Download best_model.pth from Kaggle output or your own training run
# and place it at  models/best_model.pth

# Visualise predictions on the dataset
python - <<'EOF'
import os, glob
from src.visualize import visualize_predictions

data_root = '/path/to/kaggle_3m'
image_paths = sorted(glob.glob(f'{data_root}/**/*.tif', recursive=True))
image_paths = [p for p in image_paths if 'mask' not in p]
mask_paths  = [p.replace('.tif', '_mask.tif') for p in image_paths]

visualize_predictions(
    image_paths     = image_paths,
    mask_paths      = mask_paths,
    checkpoint_path = 'models/best_model.pth',
    output_path     = 'outputs/predictions.png',
    num_samples     = 8,
)
EOF

# Run the API locally
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API reference

### `GET /`

Health check.

```json
{"status": "ok", "service": "Brain Tumour Segmentation API", "device": "cpu"}
```

### `GET /info`

Model metadata.

```json
{
  "model": "U-Net (features=[64,128,256,512])",
  "best_val_dice": 0.8586,
  "best_val_iou": 0.7645,
  "best_epoch": 45,
  "total_params": 31031233,
  "img_size": 256,
  "device": "cpu"
}
```

### `POST /predict`

Run segmentation on a single MRI slice.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_b64` | string | ✅ | Base64-encoded PNG/JPEG/TIF (RGB or grayscale) |
| `mask_b64` | string | ❌ | Base64-encoded ground-truth mask for Dice evaluation |
| `threshold` | float [0,1] | ❌ | Sigmoid threshold (default 0.5) |

**Response**

| Field | Type | Description |
|-------|------|-------------|
| `mask_b64` | string | Predicted binary mask as base64 PNG (0/255) |
| `tumour_area_pct` | float | % of pixels predicted as tumour |
| `inference_ms` | float | Inference time in milliseconds |
| `device` | string | `"cpu"` or `"cuda"` |
| `dice_score` | float \| null | Dice vs ground truth (if `mask_b64` provided) |
| `iou` | float \| null | IoU vs ground truth (if `mask_b64` provided) |

**curl example**

```bash
# Encode a TIFF slice as base64
B64=$(base64 -w0 slice.tif)

# Predict
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$B64\", \"threshold\": 0.5}" \
  | python -m json.tool
```

---

## Docker

```bash
# Build
docker build -t brain-tumour-seg:1.0 .

# Run (mount a trained model checkpoint)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  brain-tumour-seg:1.0

# Health check
curl http://localhost:8000/
```

Environment variables you can override at runtime:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/best_model.pth` | Path to checkpoint inside container |
| `IMG_SIZE` | `256` | Resize target (must match training) |
| `THRESHOLD` | `0.5` | Default sigmoid threshold |

---

## Tests

```bash
# Run all tests (dataset/API tests auto-skipped if data / model not present)
pytest tests/ -v

# Run only model + metric tests (no data needed)
pytest tests/ -v -k "Architecture or Metrics or Inference"

# With real data
DATA_PATH=/path/to/kaggle_3m pytest tests/ -v

# With trained model (enables API tests)
MODEL_PATH=models/best_model.pth pytest tests/ -v
```

Test coverage:

| Class | Tests | Data needed |
|-------|------:|-------------|
| `TestModelArchitecture` | 5 | No |
| `TestMetrics` | 8 | No |
| `TestDataset` | 3 | Yes (`DATA_PATH`) |
| `TestInferencePipeline` | 4 | No |
| `TestAPI` | 6 | Yes (`MODEL_PATH`) |

---

## Key references

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al. (2015)
- [Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm](https://doi.org/10.1016/j.compbiomed.2019.05.002) — Buda et al. (2019)
- [Kaggle LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) — mateuszbuda
