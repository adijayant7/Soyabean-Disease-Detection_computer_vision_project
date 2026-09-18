from pathlib import Path
from collections import defaultdict
import hashlib
import csv

from sklearn.model_selection import train_test_split


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "soyabean_dataset"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==========================================
# GET CLASSES
# ==========================================

classes = sorted([
    folder
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

print("=" * 60)
print("CREATING CLEAN DATASET SPLITS")
print("=" * 60)

print("\nClasses:")

for index, class_folder in enumerate(classes):
    print(f"{index}: {class_folder.name}")


# ==========================================
# CREATE HASH FUNCTION
# ==========================================

def get_file_hash(image_path):

    hasher = hashlib.md5()

    with open(image_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ==========================================
# COLLECT ONLY UNIQUE IMAGES
# ==========================================

print("\nChecking images for duplicates...")

hash_to_paths = defaultdict(list)

for class_index, class_folder in enumerate(classes):

    images = [
        file
        for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image in images:

        image_hash = get_file_hash(image)

        hash_to_paths[image_hash].append(
            (image, class_index)
        )


unique_paths = []
unique_labels = []

duplicate_count = 0
conflicting_duplicates = 0


for image_hash, items in hash_to_paths.items():

    # Get all labels belonging to this hash
    labels_for_hash = {
        label
        for path, label in items
    }

    # Same image has different labels
    if len(labels_for_hash) > 1:

        conflicting_duplicates += 1

        print(
            f"\nWARNING: Same image found "
            f"in different classes!"
        )

        for path, label in items:
            print(
                f"  {classes[label].name}: {path.name}"
            )

        # Skip conflicting images for safety
        continue


    # Keep only the first copy
    path, label = items[0]

    unique_paths.append(path)
    unique_labels.append(label)

    # Count extra copies as duplicates
    duplicate_count += len(items) - 1


print("\n" + "=" * 60)
print("DUPLICATE CLEANING RESULTS")
print("=" * 60)

print(f"Original images: {sum(len(items) for items in hash_to_paths.values())}")
print(f"Duplicate copies removed: {duplicate_count}")
print(f"Unique usable images: {len(unique_paths)}")
print(f"Conflicting duplicates skipped: {conflicting_duplicates}")


# ==========================================
# CREATE CLEAN SPLITS
# ==========================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    unique_paths,
    unique_labels,
    test_size=0.30,
    random_state=42,
    stratify=unique_labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)


# ==========================================
# SAVE SPLITS AS CSV FILES
# ==========================================

def save_split(filename, paths, labels):

    file_path = SPLIT_DIR / filename

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["image_path", "label"])

        for path, label in zip(paths, labels):

            writer.writerow([
                str(path),
                label
            ])


save_split(
    "train.csv",
    train_paths,
    train_labels
)

save_split(
    "validation.csv",
    val_paths,
    val_labels
)

save_split(
    "test.csv",
    test_paths,
    test_labels
)


# ==========================================
# SHOW RESULTS
# ==========================================

print("\n" + "=" * 60)
print("CLEAN SPLIT RESULTS")
print("=" * 60)

print(f"Training images:   {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Test images:       {len(test_paths)}")


print("\nClass distribution:")

for class_index, class_folder in enumerate(classes):

    train_count = train_labels.count(class_index)
    val_count = val_labels.count(class_index)
    test_count = test_labels.count(class_index)

    print(f"\n{class_folder.name}")
    print(f"  Train: {train_count}")
    print(f"  Validation: {val_count}")
    print(f"  Test: {test_count}")


print("\n" + "=" * 60)
print("CLEAN SPLITS SAVED SUCCESSFULLY")
print("=" * 60)

print(f"Train: {SPLIT_DIR / 'train.csv'}")
print(f"Validation: {SPLIT_DIR / 'validation.csv'}")
print(f"Test: {SPLIT_DIR / 'test.csv'}")