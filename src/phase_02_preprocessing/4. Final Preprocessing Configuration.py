import os
import random
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================
# 1. REPRODUCIBILITY
# ============================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================
# 2. DATASET PATHS
# ============================================

base_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_FINAL"

train_dir = os.path.join(base_dir, "Train")
val_dir = os.path.join(base_dir, "Validation")
test_dir = os.path.join(base_dir, "Test")


# ============================================
# 3. FINAL SETTINGS
# ============================================

IMG_SIZE = 224
BATCH_SIZE = 32

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# ============================================
# 4. FINAL TRAIN TRANSFORM
# ============================================

train_transform = transforms.Compose([

    transforms.RandomRotation(
        degrees=10,
        fill=(124, 116, 104)
    ),

    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# ============================================
# 5. FINAL VALIDATION / TEST TRANSFORM
# ============================================

eval_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# ============================================
# 6. CREATE DATASETS
# ============================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=eval_transform
)


# ============================================
# 7. CREATE DATALOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================
# 8. FINAL VERIFICATION
# ============================================

images, labels = next(iter(train_loader))

print("\n======================================")
print("FINAL PREPROCESSING CONFIGURATION")
print("======================================")

print("Random seed:", SEED)
print("Input size:", IMG_SIZE, "x", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nDataset:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

print("\nNumber of classes:", len(train_dataset.classes))

print("\nTraining augmentation:")
print("- Random rotation: ±10 degrees")
print("- Random resized crop: 0.85–1.00")
print("- Horizontal flip: p = 0.5")
print("- Brightness jitter: 0.15")
print("- Contrast jitter: 0.15")
print("- Saturation jitter: 0.10")
print("- ImageNet normalization")

print("\nValidation/Test:")
print("- Resize to 224 x 224")
print("- ImageNet normalization")
print("- No random augmentation")

print("\nBatch shape:", images.shape)

print("\nClass imbalance:")
print("Mild imbalance")
print("Standard CrossEntropyLoss will be used")
print("No class weighting required")

print("\nCUDA available:", torch.cuda.is_available())

print("\n======================================")
print("PHASE B COMPLETE ✅")
print("======================================")