import os

train_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Train"
test_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS\Classification\Test"

print("Train exists:", os.path.exists(train_dir))
print("Test exists:", os.path.exists(test_dir))

for class_name in sorted(os.listdir(train_dir)):
    train_path = os.path.join(train_dir, class_name)
    test_path = os.path.join(test_dir, class_name)

    train_count = len(os.listdir(train_path))
    test_count = len(os.listdir(test_path))

    print(
        class_name,
        "| Train:", train_count,
        "| Test:", test_count,
        "| Total:", train_count + test_count
    )