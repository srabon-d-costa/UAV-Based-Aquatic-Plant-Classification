import os
import shutil
import random
import pandas as pd
from collections import defaultdict

clean_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_CLEAN_POOL"
group_csv = r"E:\Research\Others\CVPR Aquatic plant\near_duplicate_groups.csv"

output_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_FINAL"

random.seed(42)

# Rebuild output folder safely
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

for split in ["Train", "Validation", "Test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)


# --------------------------------------------------
# Load confirmed near-duplicate groups
# --------------------------------------------------

group_df = pd.read_csv(group_csv)

group_lookup = {}

for group_id, group in group_df.groupby("group_id"):
    members = []

    for _, row in group.iterrows():
        key = (row["class"], row["file"])
        members.append(key)

    for key in members:
        group_lookup[key] = group_id


# --------------------------------------------------
# Process each class independently
# --------------------------------------------------

summary = []

classes = sorted(
    os.listdir(clean_dir),
    key=lambda x: int(x.split("_")[1])
)

for class_name in classes:

    class_path = os.path.join(clean_dir, class_name)

    files = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ]

    # Build atomic units:
    # either a near-duplicate group or a single image
    units = []
    used = set()

    # Add grouped images
    class_group_rows = group_df[group_df["class"] == class_name]

    for group_id, group in class_group_rows.groupby("group_id"):

        members = list(group["file"])

        units.append(members)

        used.update(members)

    # Add remaining single images
    for file_name in files:

        if file_name not in used:
            units.append([file_name])

    random.shuffle(units)

    total_images = len(files)

    target_train = round(total_images * 0.70)
    target_val = round(total_images * 0.15)

    train_units = []
    val_units = []
    test_units = []

    train_count = 0
    val_count = 0

    # --------------------------------------------------
    # Greedy group-aware assignment
    # --------------------------------------------------

    for unit in units:

        size = len(unit)

        if train_count + size <= target_train:
            train_units.append(unit)
            train_count += size

        elif val_count + size <= target_val:
            val_units.append(unit)
            val_count += size

        else:
            test_units.append(unit)

    split_units = {
        "Train": train_units,
        "Validation": val_units,
        "Test": test_units
    }

    split_counts = {}

    # --------------------------------------------------
    # Copy files
    # --------------------------------------------------

    for split_name, units_for_split in split_units.items():

        destination_class = os.path.join(
            output_dir,
            split_name,
            class_name
        )

        os.makedirs(destination_class, exist_ok=True)

        count = 0

        for unit in units_for_split:

            for file_name in unit:

                src = os.path.join(
                    class_path,
                    file_name
                )

                dst = os.path.join(
                    destination_class,
                    file_name
                )

                shutil.copy2(src, dst)

                count += 1

        split_counts[split_name] = count

    summary.append({
        "class": class_name,
        "total": total_images,
        "train": split_counts["Train"],
        "validation": split_counts["Validation"],
        "test": split_counts["Test"]
    })


# --------------------------------------------------
# Print summary
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

print("\n========================================")
print("GROUP-AWARE STRATIFIED SPLIT COMPLETE")
print("========================================\n")

print(summary_df.to_string(index=False))

print("\nTotals:")
print("Train:", summary_df["train"].sum())
print("Validation:", summary_df["validation"].sum())
print("Test:", summary_df["test"].sum())
print("Total:", summary_df["total"].sum())


# --------------------------------------------------
# Save split report
# --------------------------------------------------

report_path = r"E:\Research\Others\CVPR Aquatic plant\final_split_summary.csv"

summary_df.to_csv(report_path, index=False)

print("\nSplit report saved to:")
print(report_path)

print("\nFinal dataset created at:")
print(output_dir)