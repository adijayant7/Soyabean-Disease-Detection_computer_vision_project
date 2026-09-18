from pathlib import Path
from collections import defaultdict
import hashlib

from sklearn.model_selection import train_test_split


# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = Path(
    r"C:\Users\sasmi\Desktop\folders\Soyabean_Disease_Detection\data\raw\soyabean_dataset"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==========================================
# GET CLASSES
# ==========================================

classes = sorted([
    folder
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])


# ==========================================
# COLLECT IMAGES AND LABELS
# ==========================================

image_paths = []
labels = []

for class_index, class_folder in enumerate(classes):

    images = [
        file
        for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image in images:
        image_paths.append(image)
        labels.append(class_index)


# ==========================================
# CREATE SAME SPLITS AS TRAINING
# ==========================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)


# ==========================================
# CREATE IMAGE HASH
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
# HASH EACH SPLIT
# ==========================================

def get_hashes(paths):

    hashes = defaultdict(list)

    for path in paths:

        image_hash = get_file_hash(path)

        hashes[image_hash].append(path)

    return hashes


print("=" * 60)
print("CHECKING FOR EXACT DUPLICATE IMAGES")
print("=" * 60)

print("\nHashing training images...")
train_hashes = get_hashes(train_paths)

print("Hashing validation images...")
val_hashes = get_hashes(val_paths)

print("Hashing test images...")
test_hashes = get_hashes(test_paths)


# ==========================================
# CHECK OVERLAPS
# ==========================================

train_val_duplicates = (
    set(train_hashes.keys())
    & set(val_hashes.keys())
)

train_test_duplicates = (
    set(train_hashes.keys())
    & set(test_hashes.keys())
)

val_test_duplicates = (
    set(val_hashes.keys())
    & set(test_hashes.keys())
)


print("\n" + "=" * 60)
print("DUPLICATE CHECK RESULTS")
print("=" * 60)

print(
    f"\nTrain ↔ Validation duplicates: "
    f"{len(train_val_duplicates)}"
)

print(
    f"Train ↔ Test duplicates: "
    f"{len(train_test_duplicates)}"
)

print(
    f"Validation ↔ Test duplicates: "
    f"{len(val_test_duplicates)}"
)


total_duplicates = (
    len(train_val_duplicates)
    + len(train_test_duplicates)
    + len(val_test_duplicates)
)

print("\n" + "-" * 60)

if total_duplicates == 0:

    print("✓ No exact duplicate images found across splits.")

else:

    print("⚠ Exact duplicates were found across splits!")

print("-" * 60)