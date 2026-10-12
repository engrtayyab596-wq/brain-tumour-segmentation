"""
visualize.py
Visualisation utilities for brain tumour segmentation results.

Produces:
  - Side-by-side grid: MRI | Ground Truth | Prediction | Overlay
  - Single-image overlay with coloured mask
  - Training curve plot from history dict
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.model import UNet


# ─────────────────────────────────────────────────────────────
# Colour constants  (BGR-style for OpenCV overlays)
# ─────────────────────────────────────────────────────────────
TUMOUR_COLOUR = np.array([1.0, 0.2, 0.2])   # red  (RGB, 0-1)
OVERLAY_ALPHA = 0.45


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: torch.device) -> UNet:
    """Load U-Net from checkpoint dict saved by train.py."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    # checkpoint is a dict with 'model_state' key
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_val_transform(img_size: int = 256) -> A.Compose:
    """Same transform used during validation — must match dataset.py."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def _predict_single(
    model     : UNet,
    img_array : np.ndarray,         # H×W×3  uint8
    device    : torch.device,
    img_size  : int = 256,
    threshold : float = 0.5,
) -> np.ndarray:
    """
    Run model on one image.

    Returns:
        pred_mask : H×W  uint8  {0, 255}
    """
    transform = _get_val_transform(img_size)
    aug       = transform(image=img_array)
    tensor    = aug['image'].unsqueeze(0).to(device)   # 1×3×H×W

    with torch.no_grad():
        logit = model(tensor)                           # 1×1×H×W
        prob  = torch.sigmoid(logit)

    mask = (prob.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    return mask


def _colour_overlay(
    img  : np.ndarray,     # H×W×3  float32  0-1
    mask : np.ndarray,     # H×W    uint8    {0,255}
) -> np.ndarray:
    """Blend red tumour mask onto MRI image."""
    out = img.copy()
    roi = mask > 0
    out[roi] = (
        (1 - OVERLAY_ALPHA) * out[roi]
        + OVERLAY_ALPHA * TUMOUR_COLOUR
    )
    return np.clip(out, 0, 1)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def visualize_predictions(
    image_paths      : list[str],
    mask_paths       : list[str],
    checkpoint_path  : str,
    output_path      : str  = 'outputs/predictions.png',
    num_samples      : int  = 8,
    img_size         : int  = 256,
    device_str       : str  = 'auto',
) -> str:
    """
    Create a 4-column grid: MRI | Ground Truth | Prediction | Overlay

    Args:
        image_paths     : list of paths to MRI .tif files
        mask_paths      : list of paths to mask .tif files (same order)
        checkpoint_path : path to best_model.pth
        output_path     : where to save the PNG
        num_samples     : number of rows in the grid (default 8)
        img_size        : resize target (must match training)
        device_str      : 'auto' | 'cpu' | 'cuda'

    Returns:
        output_path  (for chaining)
    """
    # ── device ──────────────────────────────────────────────
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print(f"[visualize] device: {device}")

    # ── model ───────────────────────────────────────────────
    model = _load_model(checkpoint_path, device)
    print(f"[visualize] model loaded from {checkpoint_path}")

    # ── pick samples that have a tumour (more interesting) ──
    # try to pick slices with a non-empty ground-truth mask
    selected_imgs  = []
    selected_masks = []

    for img_p, msk_p in zip(image_paths, mask_paths):
        msk = np.array(Image.open(msk_p).convert('L'))
        if msk.max() > 0:
            selected_imgs.append(img_p)
            selected_masks.append(msk_p)
        if len(selected_imgs) >= num_samples:
            break

    # fallback: if not enough tumour slices, pad with any slices
    if len(selected_imgs) < num_samples:
        for img_p, msk_p in zip(image_paths, mask_paths):
            if img_p not in selected_imgs:
                selected_imgs.append(img_p)
                selected_masks.append(msk_p)
            if len(selected_imgs) >= num_samples:
                break

    n = len(selected_imgs)
    print(f"[visualize] plotting {n} samples")

    # ── figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(
        n, 4,
        figsize=(16, 4 * n),
        gridspec_kw={'wspace': 0.03, 'hspace': 0.08}
    )
    if n == 1:
        axes = axes[np.newaxis, :]   # keep 2-D

    col_titles = ['MRI Scan', 'Ground Truth', 'Prediction', 'Overlay']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=8)

    for row, (img_p, msk_p) in enumerate(zip(selected_imgs, selected_masks)):
        # load images
        img_pil = Image.open(img_p).convert('RGB')
        img_np  = np.array(img_pil)                          # H×W×3  uint8

        gt_pil  = Image.open(msk_p).convert('L')
        gt_np   = np.array(gt_pil)                           # H×W    uint8

        # resize to img_size for display
        img_disp = np.array(img_pil.resize((img_size, img_size))) / 255.0
        gt_disp  = np.array(gt_pil.resize( (img_size, img_size),
                             Image.NEAREST))

        # predict
        pred_mask = _predict_single(model, img_np, device, img_size)

        # overlay (prediction on MRI)
        overlay = _colour_overlay(img_disp, pred_mask)

        # ── column 0: MRI ──────────────────────────────────
        axes[row, 0].imshow(img_disp)

        # ── column 1: ground truth ─────────────────────────
        axes[row, 1].imshow(img_disp)
        if gt_disp.max() > 0:
            gt_colour = np.zeros((*gt_disp.shape, 4), dtype=np.float32)
            gt_colour[gt_disp > 0] = [0.2, 1.0, 0.2, 0.5]  # green
            axes[row, 1].imshow(gt_colour)

        # ── column 2: prediction ───────────────────────────
        axes[row, 2].imshow(img_disp)
        if pred_mask.max() > 0:
            pred_colour = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
            pred_colour[pred_mask > 0] = [1.0, 0.2, 0.2, 0.5]  # red
            axes[row, 2].imshow(pred_colour)

        # ── column 3: overlay ──────────────────────────────
        axes[row, 3].imshow(overlay)

        # ── y-axis label: patient + slice ─────────────────
        patient = os.path.basename(os.path.dirname(img_p))
        slice_n = os.path.splitext(os.path.basename(img_p))[0].split('_')[-1]
        axes[row, 0].set_ylabel(
            f"{patient}\nslice {slice_n}",
            fontsize=7, rotation=0,
            labelpad=70, va='center'
        )

        for col in range(4):
            axes[row, col].axis('off')

    # ── legend ──────────────────────────────────────────────
    gt_patch   = mpatches.Patch(color=(0.2, 1.0, 0.2), label='Ground truth')
    pred_patch = mpatches.Patch(color=(1.0, 0.2, 0.2), label='Prediction')
    fig.legend(
        handles=[gt_patch, pred_patch],
        loc='lower center', ncol=2,
        fontsize=11, frameon=True,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(
        'Brain Tumour Segmentation — U-Net Predictions\n'
        f'(Best val Dice = 0.8586  |  Best val IoU = 0.7645)',
        fontsize=14, fontweight='bold', y=1.002
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"[visualize] saved → {output_path}")
    return output_path


def plot_training_curves(
    history     : dict,
    output_path : str = 'outputs/training_curves.png',
) -> str:
    """
    Plot train/val loss and Dice from the history dict returned by train().

    Args:
        history     : dict with keys:
                        'train_loss', 'val_loss',
                        'train_dice', 'val_dice'
        output_path : where to save the PNG

    Returns:
        output_path
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ────────────────────────────────────────────────
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3,
             label='Train loss')
    ax1.plot(epochs, history['val_loss'],   'r-o', markersize=3,
             label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Loss')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Dice ────────────────────────────────────────────────
    ax2.plot(epochs, history['train_dice'], 'b-o', markersize=3,
             label='Train Dice')
    ax2.plot(epochs, history['val_dice'],   'r-o', markersize=3,
             label='Val Dice')

    best_epoch = int(np.argmax(history['val_dice'])) + 1
    best_dice  = max(history['val_dice'])
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    ax2.annotate(
        f'Best: {best_dice:.4f}\n(epoch {best_epoch})',
        xy=(best_epoch, best_dice),
        xytext=(best_epoch + 2, best_dice - 0.05),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='green'),
        color='green'
    )

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title('Training & Validation Dice', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.suptitle('U-Net Brain Tumour Segmentation — Training Curves',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"[visualize] training curves saved → {output_path}")
    return output_path


def overlay_single(
    image_path      : str,
    mask_path       : str | None,
    checkpoint_path : str,
    output_path     : str  = 'outputs/single_overlay.png',
    img_size        : int  = 256,
    device_str      : str  = 'auto',
) -> str:
    """
    Predict and overlay on a single MRI slice.
    Optionally draws ground-truth contour if mask_path is provided.

    Returns:
        output_path
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    model   = _load_model(checkpoint_path, device)
    img_pil = Image.open(image_path).convert('RGB')
    img_np  = np.array(img_pil)
    img_disp = np.array(img_pil.resize((img_size, img_size))) / 255.0

    pred_mask = _predict_single(model, img_np, device, img_size)
    overlay   = _colour_overlay(img_disp, pred_mask)

    cols = 3 if mask_path else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    axes[0].imshow(img_disp)
    axes[0].set_title('MRI Scan', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Prediction Overlay', fontweight='bold')
    axes[1].axis('off')

    if mask_path:
        gt_pil  = Image.open(mask_path).convert('L')
        gt_disp = np.array(gt_pil.resize((img_size, img_size), Image.NEAREST))
        axes[2].imshow(img_disp)
        if gt_disp.max() > 0:
            gt_colour = np.zeros((*gt_disp.shape, 4), dtype=np.float32)
            gt_colour[gt_disp > 0] = [0.2, 1.0, 0.2, 0.5]
            axes[2].imshow(gt_colour)
        axes[2].set_title('Ground Truth', fontweight='bold')
        axes[2].axis('off')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"[visualize] single overlay saved → {output_path}")
    return output_path
