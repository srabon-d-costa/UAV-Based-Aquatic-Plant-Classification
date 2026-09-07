import os
import matplotlib.pyplot as plt
import numpy as np

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Train"
test_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Test"

classes = sorted(
    os.listdir(train_dir),
    key=lambda x: int(x.split("_")[1])
)

train_counts = []
test_counts = []

for class_name in classes:
    train_counts.append(
        len(os.listdir(os.path.join(train_dir, class_name)))
    )

    test_counts.append(
        len(os.listdir(os.path.join(test_dir, class_name)))
    )

# Clean names for graph
clean_names = [
    c.split("_", 2)[2].replace("_", " ").title()
    for c in classes
]

x = np.arange(len(classes))
width = 0.38

plt.figure(figsize=(14, 6))

plt.bar(
    x - width/2,
    train_counts,
    width,
    label="Train"
)

plt.bar(
    x + width/2,
    test_counts,
    width,
    label="Test"
)

plt.xlabel("Aquatic Plant Species")
plt.ylabel("Number of Images")
plt.title("Class Distribution of the Aquatic Plant Dataset")

plt.xticks(
    x,
    clean_names,
    rotation=45,
    ha="right"
)

plt.legend()
plt.tight_layout()
plt.show()