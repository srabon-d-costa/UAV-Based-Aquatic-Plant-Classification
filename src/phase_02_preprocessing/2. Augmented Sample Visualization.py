import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================
# 1. DATASET PATH
# ============================================

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_FINAL\Train"


# ============================================
# 2. SETTINGS
# ============================================

IMG_SIZE = 224
BATCH_SIZE = 32

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# ============================================
# 3. FINAL TRAINING AUGMENTATION
# ============================================

train_transform = transforms.Compose([

    # Small rotation first
    transforms.RandomRotation(
        degrees=10,
        fill=(124, 116, 104)
    ),

    # Crop after rotation to reduce border artifacts
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
# 4. CREATE TRAIN DATASET
# ============================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)


# ============================================
# 5. CREATE TRAIN DATALOADER
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================
# 6. GET ONE AUGMENTED BATCH
# ============================================

images, labels = next(iter(train_loader))


# ============================================
# 7. DENORMALIZE FOR DISPLAY
# ============================================

mean_tensor = torch.tensor(
    mean
).view(3, 1, 1)

std_tensor = torch.tensor(
    std
).view(3, 1, 1)

images = images * std_tensor + mean_tensor

images = torch.clamp(
    images,
    0,
    1
)


# ============================================
# 8. VISUALIZE 12 AUGMENTED IMAGES
# ============================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(12, 9)
)

axes = axes.flatten()


for i in range(12):

    img = images[i].permute(
        1,
        2,
        0
    ).numpy()

    class_index = labels[i].item()

    class_name = train_dataset.classes[
        class_index
    ]

    clean_name = (
        class_name
        .split("_", 2)[2]
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )

    axes[i].imshow(img)

    axes[i].set_title(
        clean_name,
        fontsize=10
    )

    axes[i].axis("off")


fig.suptitle(
    "Examples of Augmented Training Images",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.show()