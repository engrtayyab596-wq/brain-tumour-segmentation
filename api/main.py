"""
api/main.py
FastAPI service for brain tumour segmentation inference.

Endpoints:
  GET  /          → health check
  GET  /info      → model metadata
  POST /predict   → base64 MRI image → predicted mask + Dice score

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker build -t brain-tumour-seg .
    docker run -p 8000:8000 brain-tumour-seg
"""

import os
import io
import base64
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model    import UNet
from src.evaluate import dice_coefficient, iou_score

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

MODEL_PATH  = os.environ.get(
    'MODEL_PATH',
    str(Path(__file__).parent.parent / 'models' / 'best_model.pth')
)
IMG_SIZE    = int(os.environ.get('IMG_SIZE', 256))
THRESHOLD   = float(os.environ.get('THRESHOLD', 0.5))
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────
# Startup — load model once at import time
# ─────────────────────────────────────────────────────────────

def _load_model() -> UNet:
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


print(f"[api] device : {DEVICE}")
print(f"[api] loading model from {MODEL_PATH}")
model = _load_model()
print(f"[api] model ready")

# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Brain Tumour Segmentation API",
    description = (
        "U-Net segmentation of LGG brain MRI slices. "
        "Submit a base64-encoded PNG/JPEG/TIF image, "
        "receive a binary tumour mask and Dice score."
    ),
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ─────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    image_b64     : str = Field(
        ...,
        description="Base64-encoded MRI image (PNG, JPEG or TIF, RGB or grayscale)"
    )
    mask_b64      : str | None = Field(
        None,
        description="Optional base64-encoded ground-truth mask for Dice evaluation"
    )
    threshold     : float = Field(
        0.5,
        ge=0.0, le=1.0,
        description="Sigmoid threshold for binarising the prediction (default 0.5)"
    )


class PredictResponse(BaseModel):
    mask_b64      : str   = Field(..., description="Predicted binary mask as base64 PNG")
    dice_score    : float | None = Field(None, description="Dice vs ground truth (if mask provided)")
    iou           : float | None = Field(None, description="IoU vs ground truth (if mask provided)")
    tumour_area_pct: float        = Field(..., description="% of pixels predicted as tumour")
    inference_ms  : float         = Field(..., description="Inference time in milliseconds")
    device        : str           = Field(..., description="Device used for inference")


class InfoResponse(BaseModel):
    model         : str
    best_val_dice : float
    best_val_iou  : float
    best_epoch    : int
    total_params  : int
    img_size      : int
    device        : str


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _b64_to_array(b64_str: str) -> np.ndarray:
    """Decode base64 string → numpy uint8 array (H×W×3 RGB)."""
    try:
        raw   = base64.b64decode(b64_str)
        img   = Image.open(io.BytesIO(raw)).convert('RGB')
        return np.array(img)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}")


def _array_to_b64(arr: np.ndarray) -> str:
    """Encode numpy array (H×W uint8) → base64 PNG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _preprocess(img_np: np.ndarray) -> torch.Tensor:
    """
    Resize + ImageNet normalise + to tensor.
    Replicates the val transform from dataset.py.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])
    return transform(image=img_np)['image'].unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the service is running."""
    return {
        "status"  : "ok",
        "service" : "Brain Tumour Segmentation API",
        "device"  : str(DEVICE),
    }


@app.get("/info", response_model=InfoResponse, tags=["Model"])
def info():
    """Return model metadata and training results."""
    total_params = sum(p.numel() for p in model.parameters())
    return InfoResponse(
        model         = "U-Net (features=[64,128,256,512])",
        best_val_dice = 0.8586,
        best_val_iou  = 0.7645,
        best_epoch    = 45,
        total_params  = total_params,
        img_size      = IMG_SIZE,
        device        = str(DEVICE),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Run tumour segmentation on a single MRI slice.

    - **image_b64**: base64-encoded MRI image (RGB or grayscale)
    - **mask_b64**: optional ground-truth mask for Dice evaluation
    - **threshold**: sigmoid threshold (default 0.5)

    Returns binary mask (base64 PNG), tumour area %, and optional Dice/IoU.
    """
    # ── decode input ────────────────────────────────────────
    img_np = _b64_to_array(request.image_b64)
    tensor = _preprocess(img_np)

    # ── inference ───────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        logit = model(tensor)            # 1×1×H×W
        prob  = torch.sigmoid(logit)     # 0-1
    inference_ms = (time.perf_counter() - t0) * 1000

    # ── binarise ────────────────────────────────────────────
    thr       = request.threshold
    pred_bin  = (prob > thr).float()                   # 1×1×H×W  {0,1}
    pred_np   = pred_bin.squeeze().cpu().numpy()       # H×W  float {0,1}
    mask_out  = (pred_np * 255).astype(np.uint8)       # H×W  uint8 {0,255}

    # ── tumour area ─────────────────────────────────────────
    tumour_pct = float(pred_np.mean() * 100)

    # ── optional Dice/IoU vs ground truth ───────────────────
    dice_val = iou_val = None
    if request.mask_b64:
        gt_raw    = base64.b64decode(request.mask_b64)
        gt_img    = Image.open(io.BytesIO(gt_raw)).convert('L').resize(
                        (IMG_SIZE, IMG_SIZE), Image.NEAREST)
        gt_np     = (np.array(gt_img) > 0).astype(np.float32)
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)  # 1×1×H×W

        dice_val = dice_coefficient(pred_bin.cpu(), gt_tensor)
        iou_val  = iou_score(pred_bin.cpu(), gt_tensor)

    return PredictResponse(
        mask_b64       = _array_to_b64(mask_out),
        dice_score     = dice_val,
        iou            = iou_val,
        tumour_area_pct= round(tumour_pct, 4),
        inference_ms   = round(inference_ms, 2),
        device         = str(DEVICE),
    )
