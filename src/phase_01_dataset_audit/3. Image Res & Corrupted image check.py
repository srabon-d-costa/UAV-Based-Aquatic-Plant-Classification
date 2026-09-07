import os
from PIL import Image
from collections import Counter

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Train"
test_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Test"

all_dirs = [train_dir, test_dir]

resolution_counter = Counter()
corrupted_files = []
total_images = 0

for base_dir in all_dirs:
    for class_name in os.listdir(base_dir):

        class_path = os.path.join(base_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):

            file_path = os.path.join(class_path, file_name)

            try:
                with Image.open(file_path) as img:

                    img.verify()

                # reopen after verify()
                with Image.open(file_path) as img:

                    width, height = img.size
                    resolution_counter[(width, height)] += 1

                total_images += 1

            except Exception as e:

                corrupted_files.append(file_path)


print("Total valid images:", total_images)
print("Corrupted images:", len(corrupted_files))
print("Number of unique resolutions:", len(resolution_counter))

print("\nMost common resolutions:")

for resolution, count in resolution_counter.most_common(10):
    print(resolution, ":", count)

if corrupted_files:
    print("\nCorrupted files:")
    for f in corrupted_files:
        print(f)