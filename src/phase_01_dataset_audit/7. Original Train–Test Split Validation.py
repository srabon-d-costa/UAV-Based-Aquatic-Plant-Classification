import os
import hashlib

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Train"
test_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Test"

def get_hash(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()

train_hashes = set()

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)

        if os.path.isfile(file_path):
            train_hashes.add(get_hash(file_path))

total_test = 0
test_found_in_train = 0

for class_name in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)

        if not os.path.isfile(file_path):
            continue

        total_test += 1

        if get_hash(file_path) in train_hashes:
            test_found_in_train += 1

print("Total Test images:", total_test)
print("Test images also found in Train:", test_found_in_train)
print(
    "Overlap percentage:",
    round((test_found_in_train / total_test) * 100, 2),
    "%"
)
