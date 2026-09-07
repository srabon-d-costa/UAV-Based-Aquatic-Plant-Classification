import os
import matplotlib.pyplot as plt
from PIL import Image

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Train"

classes = sorted(
    os.listdir(train_dir),
    key=lambda x: int(x.split("_")[1])
)

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, class_name in enumerate(classes):

    class_path = os.path.join(train_dir, class_name)

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_path = os.path.join(class_path, image_files[0])

    img = Image.open(image_path).convert("RGB")

    axes[i].imshow(img)

    clean_name = (
        class_name
        .split("_", 2)[2]
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )

    axes[i].set_title(
        clean_name,
        fontsize=10,
        pad=8
    )

    axes[i].axis("off")


# Remove unused cells
for j in range(len(classes), len(axes)):
    axes[j].axis("off")


fig.suptitle(
    "Representative Samples from the 14 Aquatic Plant Classes",
    fontsize=18,
    y=0.98
)

# Better spacing
plt.subplots_adjust(
    top=0.91,
    bottom=0.05,
    left=0.04,
    right=0.98,
    hspace=0.38,
    wspace=0.20
)

plt.show()