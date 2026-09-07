import os
import pandas as pd
from collections import defaultdict

csv_path = r"E:\Research\Others\CVPR Aquatic plant\verified_near_duplicates.csv"

df = pd.read_csv(csv_path)

# Keep only strongly verified near-duplicates
df = df[df["ssim"] >= 0.98].copy()

print("Confirmed near-duplicate pairs:", len(df))


# ----------------------------------------
# Union-Find for grouping connected images
# ----------------------------------------

parent = {}


def find(x):

    if x not in parent:
        parent[x] = x

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# ----------------------------------------
# Build groups separately within classes
# ----------------------------------------

for _, row in df.iterrows():

    class_name = row["class"]

    image1 = f"{class_name}/{row['image_1']}"
    image2 = f"{class_name}/{row['image_2']}"

    union(image1, image2)


# ----------------------------------------
# Collect groups
# ----------------------------------------

groups = defaultdict(list)

for image in parent:

    root = find(image)

    groups[root].append(image)


near_duplicate_groups = [
    images
    for images in groups.values()
    if len(images) > 1
]


print("\n===================================")
print("NEAR-DUPLICATE GROUPING")
print("===================================")

print(
    "Confirmed pairs:",
    len(df)
)

print(
    "Number of near-duplicate groups:",
    len(near_duplicate_groups)
)

print(
    "Images involved in groups:",
    len(
        set(
            image
            for group in near_duplicate_groups
            for image in group
        )
    )
)


for i, group in enumerate(
    near_duplicate_groups,
    start=1
):

    print(f"\nGroup {i}:")

    for image in group:
        print("   ", image)


# ----------------------------------------
# Save group report
# ----------------------------------------

rows = []

for group_id, group in enumerate(
    near_duplicate_groups,
    start=1
):

    for image in group:

        class_name, file_name = image.split("/", 1)

        rows.append({
            "group_id": group_id,
            "class": class_name,
            "file": file_name
        })


group_df = pd.DataFrame(rows)

output = (
    r"E:\Research\Others\CVPR Aquatic plant"
    r"\near_duplicate_groups.csv"
)

group_df.to_csv(
    output,
    index=False
)

print("\nGroup report saved to:")
print(output)