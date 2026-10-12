import torch


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    # apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # apply threshold to get binary mask
    pred = (pred > threshold).float()

    # flatten both tensors to 1D for calculation
    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def dice_loss(pred, target, smooth=1e-6):
    # used during training — works on probabilities not binary
    # no threshold applied so gradients can flow
    pred = torch.sigmoid(pred)

    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1.0 - dice
