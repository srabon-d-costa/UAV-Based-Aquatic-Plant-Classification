import os
from PIL import Image
import imagehash

clean_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_CLEAN_POOL"

image_records = []

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Step 1: calculate perceptual hashes
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

                image_records.append({
                    "class": class_name,
                    "file": file_name,
                    "path": file_path,
                    "hash": phash
                })

        except Exception as e:
            print("Error:", file_path, e)


print("Images successfully hashed:", len(image_records))


# Step 2: compare images
near_duplicates = []

# Smaller distance = more visually similar
THRESHOLD = 4

for i in range(len(image_records)):

    for j in range(i + 1, len(image_records)):

        # Only compare images from the same class
        if image_records[i]["class"] != image_records[j]["class"]:
            continue

        distance = (
            image_records[i]["hash"]
            -
            image_records[j]["hash"]
        )

        if distance <= THRESHOLD:

            near_duplicates.append({
                "class": image_records[i]["class"],
                "image_1": image_records[i]["file"],
                "image_2": image_records[j]["file"],
                "distance": distance
            })


print("\n==============================")
print("NEAR-DUPLICATE AUDIT")
print("==============================")

print("Total images checked:", len(image_records))
print("Near-duplicate pairs found:", len(near_duplicates))


print("\nFirst 50 near-duplicate pairs:")

for pair in near_duplicates[:50]:

    print(
        pair["class"],
        "|",
        pair["image_1"],
        "<->",
        pair["image_2"],
        "| pHash distance:",
        pair["distance"]
    )