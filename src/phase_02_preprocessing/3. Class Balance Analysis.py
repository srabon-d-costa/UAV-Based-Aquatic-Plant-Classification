import os
import pandas as pd
import matplotlib.pyplot as plt

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_FINAL\Train"

# ============================================
# 1. COUNT TRAINING IMAGES PER CLASS
# ============================================

class_counts = {}

for class_name in sorted(
    os.listdir(train_dir),
    key=lambda x: int(x.split("_")[1])
):
    class_path = os.path.join(train_dir, class_name)

    if os.path.isdir(class_path):
        count = len([
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ])

        class_counts[class_name] = count


# ============================================
# 2. CREATE TABLE
# ============================================

df = pd.DataFrame(
    list(class_counts.items()),
    columns=["Class", "Train Images"]
)

print("\n===================================")
print("TRAINING CLASS BALANCE ANALYSIS")
print("===================================\n")

print(df.to_string(index=False))

total = df["Train Images"].sum()

minimum = df["Train Images"].min()
maximum = df["Train Images"].max()

imbalance_ratio = maximum / minimum

print("\nTotal training images:", total)
print("Smallest class:", minimum)
print("Largest class:", maximum)
print("Imbalance ratio:", round(imbalance_ratio, 3))


# ============================================
# 3. CLEAN NAMES FOR GRAPH
# ============================================

clean_names = [
    name.split("_", 2)[2]
    .replace("_", " ")
    .replace("-", " ")
    .title()
    for name in df["Class"]
]


# ============================================
# 4. CLASS DISTRIBUTION GRAPH
# ============================================

plt.figure(figsize=(14, 6))

plt.bar(
    clean_names,
    df["Train Images"]
)

plt.xlabel("Aquatic Plant Species")
plt.ylabel("Number of Training Images")

plt.title(
    "Class Distribution of the Final Leakage-Free Training Set"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================
# 5. SIMPLE CLASS-WEIGHT DECISION
# ============================================

print("\n===================================")
print("CLASS WEIGHT DECISION")
print("===================================")

if imbalance_ratio <= 1.5:
    print("Class imbalance is mild.")
    print("Weighted loss is NOT necessary.")
else:
    print("Class imbalance is notable.")
    print("Class-weighted loss may be useful.")
    