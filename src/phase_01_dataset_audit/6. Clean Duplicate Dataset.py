import os
import shutil
import hashlib
from collections import defaultdict

# Original dataset
train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Train"
test_dir  = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Test"

# New clean dataset folder
output_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_CLEAN_POOL"

os.makedirs(output_dir, exist_ok=True)

def file_hash(path):
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


seen_hashes = {}
class_counts = defaultdict(int)
duplicate_count = 0
unique_count = 0

source_dirs = [train_dir, test_dir]

for source_dir in source_dirs:

    for class_name in sorted(os.listdir(source_dir)):

        class_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        clean_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(clean_class_dir, exist_ok=True)

        for file_name in os.listdir(class_path):

            file_path = os.path.join(class_path, file_name)

            if not os.path.isfile(file_path):
                continue

            try:
                h = file_hash(file_path)

                # Duplicate already encountered
                if h in seen_hashes:
                    duplicate_count += 1
                    continue

                # First occurrence
                seen_hashes[h] = file_path

                destination = os.path.join(
                    clean_class_dir,
                    file_name
                )

                # Avoid accidental filename collision
                if os.path.exists(destination):

                    name, ext = os.path.splitext(file_name)

                    destination = os.path.join(
                        clean_class_dir,
                        f"{name}_{unique_count}{ext}"
                    )

                shutil.copy2(
                    file_path,
                    destination
                )

                class_counts[class_name] += 1
                unique_count += 1

            except Exception as e:
                print("Error:", file_path, e)


print("\n==============================")
print("DEDUPLICATION COMPLETE")
print("==============================")

print("Unique images:", unique_count)
print("Exact duplicates removed:", duplicate_count)

print("\nUnique images per class:")

for class_name in sorted(
    class_counts,
    key=lambda x: int(x.split("_")[1])
):
    print(
        class_name,
        ":",
        class_counts[class_name]
    )

print("\nClean dataset created at:")
print(output_dir)