import os
import hashlib
import pandas as pd
from collections import defaultdict

final_dir = r"E:\Research\Others\CVPR Aquatic plant\AquPlantDS_FINAL"
group_csv = r"E:\Research\Others\CVPR Aquatic plant\near_duplicate_groups.csv"

splits = ["Train", "Validation", "Test"]


# ==================================================
# PART 1 — EXACT DUPLICATE CHECK ACROSS SPLITS
# ==================================================

def get_md5(path):
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


hash_records = defaultdict(list)

for split in splits:

    split_path = os.path.join(final_dir, split)

    for class_name in os.listdir(split_path):

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):

            file_path = os.path.join(
                class_path,
                file_name
            )

            if not os.path.isfile(file_path):
                continue

            h = get_md5(file_path)

            hash_records[h].append({
                "split": split,
                "class": class_name,
                "file": file_name
            })


exact_cross_split_duplicates = []

for h, records in hash_records.items():

    split_names = set(
        r["split"] for r in records
    )

    if len(split_names) > 1:
        exact_cross_split_duplicates.append(
            (h, records)
        )


print("\n======================================")
print("1. EXACT CROSS-SPLIT DUPLICATE CHECK")
print("======================================")

print(
    "Exact duplicate hashes across splits:",
    len(exact_cross_split_duplicates)
)

if exact_cross_split_duplicates:

    for h, records in exact_cross_split_duplicates:

        print("\nDuplicate group:")

        for r in records:
            print(
                r["split"],
                "|",
                r["class"],
                "|",
                r["file"]
            )


# ==================================================
# PART 2 — NEAR-DUPLICATE GROUP SPLIT CHECK
# ==================================================

group_df = pd.read_csv(group_csv)

# Build lookup:
# (class, filename) -> split
file_to_split = {}

for split in splits:

    split_path = os.path.join(final_dir, split)

    for class_name in os.listdir(split_path):

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):

            file_to_split[
                (class_name, file_name)
            ] = split


group_violations = []

for group_id, group in group_df.groupby("group_id"):

    locations = []

    for _, row in group.iterrows():

        key = (
            row["class"],
            row["file"]
        )

        split = file_to_split.get(
            key,
            "NOT FOUND"
        )

        locations.append({
            "class": row["class"],
            "file": row["file"],
            "split": split
        })

    split_set = set(
        item["split"]
        for item in locations
    )

    if len(split_set) > 1:

        group_violations.append({
            "group_id": group_id,
            "locations": locations
        })


print("\n======================================")
print("2. NEAR-DUPLICATE GROUP CHECK")
print("======================================")

print(
    "Near-duplicate groups split across partitions:",
    len(group_violations)
)

if group_violations:

    for violation in group_violations:

        print(
            "\nGroup",
            violation["group_id"]
        )

        for item in violation["locations"]:

            print(
                item["split"],
                "|",
                item["class"],
                "|",
                item["file"]
            )


# ==================================================
# FINAL VERDICT
# ==================================================

print("\n======================================")
print("FINAL LEAKAGE VERIFICATION")
print("======================================")

if (
    len(exact_cross_split_duplicates) == 0
    and
    len(group_violations) == 0
):

    print("PASS ✅")
    print(
        "No exact duplicates were found across "
        "Train / Validation / Test."
    )

    print(
        "No confirmed near-duplicate group was "
        "split across different partitions."
    )

    print(
        "The final dataset split is leakage-free "
        "under the performed duplicate audits."
    )

else:

    print("FAIL ❌")

    print(
        "Leakage issues remain and should be fixed "
        "before model training."
    )