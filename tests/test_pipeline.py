"""
tests/test_pipeline.py
pytest suite for the brain tumour segmentation pipeline.

Tests:
  1. Model architecture — forward pass shape
  2. Metric functions — perfect / wrong prediction
  3. Dataset loading — paths, shapes, mask values
  4. Full inference pipeline — end-to-end predict + Dice
  5. API endpoints — health, info, predict (via TestClient)

Run:
    pytest tests/ -v
"""

import io
import os
import sys
import base64
import pytest
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model    import UNet
from src.evaluate import dice_loss, dice_coefficient, iou_score

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def device():
    return torch.device('cpu')        # always use CPU for unit tests


@pytest.fixture(scope='module')
def unet_model(device):
    """Fresh (untrained) U-Net for shape tests."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.eval()
    return model


@pytest.fixture(scope='module')
def dummy_batch(device):
    """Batch of 2 random images — (B,3,H,W) and (B,1,H,W) masks."""
    imgs  = torch.rand(2, 3, 256, 256, device=device)
    masks = torch.randint(0, 2, (2, 1, 256, 256),
                          dtype=torch.float32, device=device)
    return imgs, masks


@pytest.fixture(scope='module')
def dummy_image_b64() -> str:
    """Tiny 64×64 white RGB image as base64 PNG."""
    arr = np.ones((64, 64, 3), dtype=np.uint8) * 128
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ─────────────────────────────────────────────────────────────
# 1. Model architecture
# ─────────────────────────────────────────────────────────────

class TestModelArchitecture:

    def test_output_shape(self, unet_model, dummy_batch, device):
        """U-Net output must be (B, 1, H, W) — same spatial dims as input."""
        imgs, _ = dummy_batch
        with torch.no_grad():
            out = unet_model(imgs)
        assert out.shape == (2, 1, 256, 256), (
            f"Expected (2,1,256,256), got {out.shape}"
        )

    def test_output_range_pre_sigmoid(self, unet_model, dummy_batch):
        """Raw logits should NOT all be in [0,1] (they are unbounded)."""
        imgs, _ = dummy_batch
        with torch.no_grad():
            out = unet_model(imgs)
        # at least some logits should be outside [0,1]
        outside = ((out < 0) | (out > 1)).any()
        assert outside, "Logits appear bounded — did sigmoid leak into model?"

    def test_parameter_count(self, unet_model):
        """U-Net with features=[64,128,256,512] should have ~31M params."""
        n = sum(p.numel() for p in unet_model.parameters())
        assert 28_000_000 < n < 35_000_000, (
            f"Unexpected parameter count: {n:,}"
        )

    def test_single_image_input(self, unet_model, device):
        """Model must handle batch size 1."""
        x = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out = unet_model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_no_nan_in_output(self, unet_model, dummy_batch):
        """No NaN or Inf in forward pass with random input."""
        imgs, _ = dummy_batch
        with torch.no_grad():
            out = unet_model(imgs)
        assert not torch.isnan(out).any(), "NaN in model output"
        assert not torch.isinf(out).any(), "Inf in model output"


# ─────────────────────────────────────────────────────────────
# 2. Metric functions
# ─────────────────────────────────────────────────────────────

class TestMetrics:

    def _ones(self, shape=(1, 1, 8, 8)):
        return torch.ones(shape)

    def _zeros(self, shape=(1, 1, 8, 8)):
        return torch.zeros(shape)

    def test_dice_loss_perfect(self):
        """Dice loss = 0.0 when pred == target (all ones)."""
        pred = torch.sigmoid(torch.ones(1, 1, 8, 8) * 10)  # prob ≈ 1
        target = self._ones()
        loss = dice_loss(pred, target)
        assert loss < 0.01, f"Expected ~0, got {loss:.4f}"

    def test_dice_loss_worst(self):
        """Dice loss ≈ 1.0 when pred is all zeros vs all-ones target."""
        pred   = torch.sigmoid(torch.ones(1, 1, 8, 8) * -10)  # prob ≈ 0
        target = self._ones()
        loss   = dice_loss(pred, target)
        assert loss > 0.98, f"Expected ~1, got {loss:.4f}"

    def test_dice_coefficient_perfect(self):
        """Dice coefficient = 1.0 for identical binary masks."""
        pred   = self._ones()   # already 0/1
        target = self._ones()
        dc = dice_coefficient(pred, target)
        assert abs(dc - 1.0) < 1e-4, f"Expected 1.0, got {dc:.4f}"

    def test_dice_coefficient_no_overlap(self):
        """Dice coefficient = 0.0 for completely different masks."""
        pred   = self._ones()
        target = self._zeros()
        dc = dice_coefficient(pred, target)
        assert abs(dc - 0.0) < 1e-4, f"Expected 0.0, got {dc:.4f}"

    def test_iou_perfect(self):
        """IoU = 1.0 for identical masks."""
        pred   = self._ones()
        target = self._ones()
        iou = iou_score(pred, target)
        assert abs(iou - 1.0) < 1e-4, f"Expected 1.0, got {iou:.4f}"

    def test_iou_no_overlap(self):
        """IoU = 0.0 for non-overlapping masks."""
        pred   = self._ones()
        target = self._zeros()
        iou = iou_score(pred, target)
        assert abs(iou - 0.0) < 1e-4, f"Expected 0.0, got {iou:.4f}"

    def test_dice_range(self, dummy_batch):
        """Dice coefficient must always be in [0, 1]."""
        _, masks = dummy_batch
        pred = torch.rand_like(masks)   # random probs
        dc   = dice_coefficient(pred, masks)
        assert 0.0 <= dc <= 1.0, f"Dice out of range: {dc}"

    def test_iou_range(self, dummy_batch):
        """IoU must always be in [0, 1]."""
        _, masks = dummy_batch
        pred = torch.rand_like(masks)
        iou  = iou_score(pred, masks)
        assert 0.0 <= iou <= 1.0, f"IoU out of range: {iou}"


# ─────────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────────

class TestDataset:

    DATA_PATH = os.environ.get(
        'DATA_PATH',
        '/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m'
    )

    @pytest.mark.skipif(
        not os.path.exists(DATA_PATH),
        reason="LGG dataset not available in this environment"
    )
    def test_dataset_length(self):
        """Dataset should have ~3929 image/mask pairs."""
        from src.dataset import BrainMRIDataset
        ds = BrainMRIDataset.__new__(BrainMRIDataset)
        # count pairs manually
        images, masks = [], []
        for patient in os.listdir(self.DATA_PATH):
            p = os.path.join(self.DATA_PATH, patient)
            if not os.path.isdir(p):
                continue
            for f in os.listdir(p):
                if f.endswith('.tif') and 'mask' not in f:
                    img_p = os.path.join(p, f)
                    msk_p = img_p.replace('.tif', '_mask.tif')
                    if os.path.exists(msk_p):
                        images.append(img_p)
                        masks.append(msk_p)
        assert len(images) > 3000, f"Too few pairs: {len(images)}"

    @pytest.mark.skipif(
        not os.path.exists(DATA_PATH),
        reason="LGG dataset not available in this environment"
    )
    def test_dataset_item_shapes(self):
        """Each item must return (3,256,256) image and (1,256,256) mask."""
        from src.dataset import BrainMRIDataset
        images, masks = [], []
        for patient in sorted(os.listdir(self.DATA_PATH))[:5]:
            p = os.path.join(self.DATA_PATH, patient)
            if not os.path.isdir(p):
                continue
            for f in os.listdir(p):
                if f.endswith('.tif') and 'mask' not in f:
                    img_p = os.path.join(p, f)
                    msk_p = img_p.replace('.tif', '_mask.tif')
                    if os.path.exists(msk_p):
                        images.append(img_p)
                        masks.append(msk_p)
                        break

        ds = BrainMRIDataset(images[:3], masks[:3], augment=False)
        img, msk = ds[0]
        assert img.shape == (3, 256, 256), f"Image shape wrong: {img.shape}"
        assert msk.shape == (1, 256, 256), f"Mask shape wrong: {msk.shape}"

    @pytest.mark.skipif(
        not os.path.exists(DATA_PATH),
        reason="LGG dataset not available in this environment"
    )
    def test_mask_binary_values(self):
        """Mask tensor must only contain 0.0 and 1.0."""
        from src.dataset import BrainMRIDataset
        images, masks = [], []
        for patient in sorted(os.listdir(self.DATA_PATH))[:3]:
            p = os.path.join(self.DATA_PATH, patient)
            if not os.path.isdir(p):
                continue
            for f in os.listdir(p):
                if f.endswith('.tif') and 'mask' not in f:
                    img_p = os.path.join(p, f)
                    msk_p = img_p.replace('.tif', '_mask.tif')
                    if os.path.exists(msk_p):
                        images.append(img_p)
                        masks.append(msk_p)
                        break

        ds = BrainMRIDataset(images[:3], masks[:3], augment=False)
        _, msk = ds[0]
        unique = torch.unique(msk).tolist()
        for v in unique:
            assert v in (0.0, 1.0), f"Unexpected mask value: {v}"


# ─────────────────────────────────────────────────────────────
# 4. Full inference pipeline
# ─────────────────────────────────────────────────────────────

class TestInferencePipeline:

    def test_predict_output_shape(self, unet_model, device):
        """End-to-end: random image → sigmoid → threshold → correct shape."""
        x     = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            logit = unet_model(x)
            prob  = torch.sigmoid(logit)
            pred  = (prob > 0.5).float()

        assert pred.shape == (1, 1, 256, 256)

    def test_predict_binary_output(self, unet_model, device):
        """After thresholding, output must only contain 0 and 1."""
        x = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            prob = torch.sigmoid(unet_model(x))
            pred = (prob > 0.5).float()

        unique = torch.unique(pred).tolist()
        for v in unique:
            assert v in (0.0, 1.0), f"Unexpected value after threshold: {v}"

    def test_sigmoid_output_range(self, unet_model, device):
        """Sigmoid output must be strictly in [0, 1]."""
        x = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            prob = torch.sigmoid(unet_model(x))
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_different_inputs_different_outputs(self, unet_model, device):
        """Two different inputs must produce different outputs."""
        x1 = torch.zeros(1, 3, 256, 256, device=device)
        x2 = torch.ones( 1, 3, 256, 256, device=device)
        with torch.no_grad():
            o1 = torch.sigmoid(unet_model(x1))
            o2 = torch.sigmoid(unet_model(x2))
        assert not torch.allclose(o1, o2), "Model output is constant"


# ─────────────────────────────────────────────────────────────
# 5. API endpoints (TestClient — no server needed)
# ─────────────────────────────────────────────────────────────

# Only run API tests if a trained model file exists
_MODEL_EXISTS = os.path.exists(
    os.environ.get('MODEL_PATH', 'models/best_model.pth')
)


@pytest.mark.skipif(
    not _MODEL_EXISTS,
    reason="Trained model file not found — skipping API tests"
)
class TestAPI:

    @pytest.fixture(scope='class')
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health_check(self, client):
        """GET / → 200 with status=ok."""
        r = client.get('/')
        assert r.status_code == 200
        data = r.json()
        assert data['status'] == 'ok'

    def test_info_endpoint(self, client):
        """GET /info → correct model metadata."""
        r = client.get('/info')
        assert r.status_code == 200
        data = r.json()
        assert data['best_val_dice'] == pytest.approx(0.8586, abs=1e-3)
        assert data['img_size'] == 256

    def test_predict_returns_mask(self, client, dummy_image_b64):
        """POST /predict → 200 with mask_b64 and tumour_area_pct."""
        payload = {'image_b64': dummy_image_b64, 'threshold': 0.5}
        r = client.post('/predict', json=payload)
        assert r.status_code == 200
        data = r.json()
        assert 'mask_b64' in data
        assert 'tumour_area_pct' in data
        assert 'inference_ms' in data
        assert data['inference_ms'] > 0

    def test_predict_mask_decodable(self, client, dummy_image_b64):
        """Returned mask_b64 must decode to a valid PNG image."""
        payload = {'image_b64': dummy_image_b64}
        r = client.post('/predict', json=payload)
        mask_bytes = base64.b64decode(r.json()['mask_b64'])
        img = Image.open(io.BytesIO(mask_bytes))
        assert img is not None

    def test_predict_invalid_image(self, client):
        """POST /predict with garbage b64 → 422."""
        payload = {'image_b64': 'not_valid_base64!!!'}
        r = client.post('/predict', json=payload)
        assert r.status_code == 422

    def test_predict_with_gt_mask(self, client, dummy_image_b64):
        """POST /predict with mask_b64 → dice_score and iou returned."""
        # use same image as both input and ground-truth mask
        payload = {
            'image_b64' : dummy_image_b64,
            'mask_b64'  : dummy_image_b64,   # not a real mask but parseable
            'threshold' : 0.5,
        }
        r = client.post('/predict', json=payload)
        assert r.status_code == 200
        data = r.json()
        # dice and iou should be present (may be 0 since it's a random mask)
        assert 'dice_score' in data
        assert 'iou' in data
