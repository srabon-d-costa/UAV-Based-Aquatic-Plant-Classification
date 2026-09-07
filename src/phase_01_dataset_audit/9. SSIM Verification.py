import os
import imagehash
import pandas as pd
import numpy as np

from PIL import Image
from skimage.metrics import structural_similarity as ssim


clean_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_CLEAN_POOL"

records = []

valid_extensions = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)


# 1. Generate pHash for all clean images
for class_name in sorted(os.listdir(clean_dir)):

    class_path = os.path.join(clean_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):

        if not file_name.lower().endswith(valid_extensions):
            continue

        file_path = os.path.join(class_path, file_name)

        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                phash = imagehash.phash(img)

            records.append({
                "class": class_name,
                "file": file_name,
                "path": file_path,
                "phash": phash
            })

        except Exception as e:
            print("Error:", file_path, e)


print("Images loaded:", len(records))


# 2. SSIM function
def calculate_ssim(path1, path2):

    img1 = Image.open(path1).convert("L")
    img2 = Image.open(path2).convert("L")

    # Same dimensions for SSIM
    img1 = img1.resize((256, 256))
    img2 = img2.resize((256, 256))

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    return ssim(
        arr1,
        arr2,
        data_range=255
    )


# 3. pHash candidate detection + SSIM verification
PHASH_THRESHOLD = 4

results = []

for i in range(len(records)):

    for j in range(i + 1, len(records)):

        # Only compare within same species
        if records[i]["class"] != records[j]["class"]:
            continue

        phash_distance = (
            records[i]["phash"]
            -
            records[j]["phash"]
        )

        if phash_distance <= PHASH_THRESHOLD:

            similarity = calculate_ssim(
                records[i]["path"],
                records[j]["path"]
            )

            results.append({
                "class": records[i]["class"],
                "image_1": records[i]["file"],
                "image_2": records[j]["file"],
                "phash_distance": phash_distance,
                "ssim": similarity
            })


# 4. Save report
df = pd.DataFrame(results)

output_csv = (
    r"E:\Research\Others\CVPR Aquatic plant"
    r"\verified_near_duplicates.csv"
)

df.to_csv(output_csv, index=False)


# 5. Summary
very_high = df[df["ssim"] >= 0.98]

high = df[
    (df["ssim"] >= 0.95)
    &
    (df["ssim"] < 0.98)
]


print("\n===================================")
print("NEAR-DUPLICATE VERIFICATION")
print("===================================")

print("pHash candidate pairs:", len(df))
print("SSIM >= 0.98:", len(very_high))
print("SSIM 0.95-0.98:", len(high))

print("\nMost similar pairs:")

print(
    df.sort_values(
        "ssim",
        ascending=False
    ).head(30).to_string(index=False)
)

print("\nCSV saved to:")
print(output_csv)