import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BrainMRIDataset(Dataset):
    """
    Dataset class for brain MRI segmentation.
    Loads image-mask pairs and applies augmentation during training.
    """

    def __init__(self, image_paths, mask_paths,
                 img_size=256, augment=False):
        """
        Args:
            image_paths : list of paths to MRI slice images
            mask_paths  : list of paths to corresponding masks
            img_size    : resize all images to this size
            augment     : apply data augmentation if True
                          use True for training, False for validation
        """
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.img_size    = img_size
        self.augment     = augment
        self.transform   = self._build_transforms()

    def _build_transforms(self):
        if self.augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.4),
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05,
                    p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.4
                ),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load mask as grayscale
        mask = cv2.imread(self.mask_paths[idx],
                          cv2.IMREAD_GRAYSCALE)

        # convert mask to binary 0 or 1
        mask = (mask > 0).astype(np.float32)

        # apply transforms
        transformed = self.transform(image=img, mask=mask)
        img  = transformed['image']
        mask = transformed['mask']

        # mask needs channel dimension (1, H, W)
        mask = mask.unsqueeze(0)

        return img, mask
