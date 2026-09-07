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
# 3. IMAGE SETTINGS
# ============================================

IMG_SIZE = 224
BATCH_SIZE = 32

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# ============================================
# 4. TRAINING TRANSFORMS
# ============================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
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
# 5. VALIDATION / TEST TRANSFORMS
# ============================================

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

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
# 8. DATASET SUMMARY
# ============================================

print("\n====================================")
print("PHASE B — DATA PIPELINE VERIFICATION")
print("====================================")

print("\nTrain images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

print("\nNumber of classes:", len(train_dataset.classes))

# ============================================
# 9. CLASS MAPPING
# ============================================

print("\nClass Mapping:")

for class_name, class_index in train_dataset.class_to_idx.items():
    print(
        class_index,
        "->",
        class_name
    )

# ============================================
# 10. CHECK ONE TRAINING BATCH
# ============================================

images, labels = next(iter(train_loader))

print("\nBatch verification:")
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("\nExample labels:")
print(labels[:10])

# ============================================
# 11. BASIC CONSISTENCY CHECKS
# ============================================

assert len(train_dataset) == 744
assert len(val_dataset) == 161
assert len(test_dataset) == 157

assert train_dataset.classes == val_dataset.classes
assert train_dataset.classes == test_dataset.classes

assert images.shape[1] == 3
assert images.shape[2] == 224
assert images.shape[3] == 224

print("\nAll checks passed ✅")