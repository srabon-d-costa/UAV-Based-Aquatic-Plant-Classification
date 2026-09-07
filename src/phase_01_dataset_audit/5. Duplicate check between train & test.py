import os
import hashlib

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Train"
test_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Test"


def get_file_hash(file_path):
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


train_hashes = {}
test_hashes = {}

# Train hashes
for class_name in os.listdir(train_dir):

    class_path = os.path.join(train_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        try:
            file_hash = get_file_hash(file_path)

            train_hashes[file_hash] = {
                "class": class_name,
                "file": file_name,
                "path": file_path
            }

        except:
            pass


# Test hashes
for class_name in os.listdir(test_dir):

    class_path = os.path.join(test_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        try:
            file_hash = get_file_hash(file_path)

            test_hashes[file_hash] = {
                "class": class_name,
                "file": file_name,
                "path": file_path
            }

        except:
            pass


duplicate_hashes = set(train_hashes.keys()) & set(test_hashes.keys())

print("Unique train images:", len(train_hashes))
print("Unique test images:", len(test_hashes))
print("Exact Train-Test duplicates:", len(duplicate_hashes))


if duplicate_hashes:

    print("\nDuplicate pairs:")

    for h in duplicate_hashes:

        print("\nTRAIN:")
        print(train_hashes[h]["class"])
        print(train_hashes[h]["file"])

        print("TEST:")
        print(test_hashes[h]["class"])
        print(test_hashes[h]["file"])