import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset  import BrainMRIDataset
from src.model    import UNet
from src.evaluate import dice_coefficient, iou_score, dice_loss


def train_one_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimiser.zero_grad()
        preds = model(imgs)
        loss  = dice_loss(preds, masks)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(preds.detach(), masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_dice


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            total_loss += dice_loss(preds, masks).item()
            total_dice += dice_coefficient(preds, masks)
            total_iou  += iou_score(preds, masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou  = total_iou  / len(loader)

    return avg_loss, avg_dice, avg_iou


def train(image_paths, mask_paths,
          epochs       = 30,
          batch_size   = 8,
          lr           = 1e-4,
          img_size     = 256,
          save_path    = 'models/best_model.pth',
          device       = None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'cpu')

    print(f"Training on: {device}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch_size}")
    print(f"LR:          {lr}")
    print()

    # split dataset
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        test_size=0.2, random_state=42
    )

    print(f"Train samples: {len(train_imgs)}")
    print(f"Val samples:   {len(val_imgs)}")

    # datasets and loaders
    train_dataset = BrainMRIDataset(train_imgs, train_masks,
                                     img_size=img_size, augment=True)
    val_dataset   = BrainMRIDataset(val_imgs, val_masks,
                                     img_size=img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # model, optimiser, scheduler
    model     = UNet(in_channels=3, out_channels=1).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', patience=5, factor=0.5
    )

    best_dice    = 0.0
    history      = []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimiser, device
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, device
        )

        scheduler.step(val_loss)

        history.append({
            'epoch'     : epoch,
            'train_loss': round(train_loss, 4),
            'train_dice': round(train_dice, 4),
            'val_loss'  : round(val_loss,   4),
            'val_dice'  : round(val_dice,   4),
            'val_iou'   : round(val_iou,    4),
        })

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_dice: {train_dice:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_dice: {val_dice:.4f} | "
              f"val_iou: {val_iou:.4f}")

        # save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'val_dice'  : val_dice,
                'val_iou'   : val_iou,
            }, save_path)
            print(f"  → best model saved (val_dice={val_dice:.4f})")

    print(f"\nTraining complete. Best val_dice: {best_dice:.4f}")
    print(f"Model saved to: {save_path}")

    return model, history
