from pathlib import Path
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(
    r"C:\Users\sasmi\Desktop\folders\Soyabean_Disease_Detection\data\raw\soyabean_dataset"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}



if not DATASET_PATH.exists():
    print(f"Dataset not found: {DATASET_PATH}")
    exit()



classes = sorted([
    folder
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])


print("=" * 50)
print("DATASET SPLITTING")
print("=" * 50)

print("\nClasses:")

for i, class_folder in enumerate(classes):
    print(f"{i}: {class_folder.name}")



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


print(f"\nTotal images: {len(image_paths)}")



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



print("\n" + "=" * 50)
print("SPLIT RESULTS")
print("=" * 50)

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
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")

from pathlib import Path
from sklearn.model_selection import train_test_split
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

def save_split(file_name, paths, labels):

    output_file = PROCESSED_PATH / file_name

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["image_path", "label"])

        for image_path, label in zip(paths, labels):

            relative_path = image_path.relative_to(PROJECT_ROOT)

            writer.writerow([relative_path, label])


PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

save_split(
    "train.csv",
    train_paths,
    train_labels
)

save_split(
    "val.csv",
    val_paths,
    val_labels
)

save_split(
    "test.csv",
    test_paths,
    test_labels
)

print("\n" + "=" * 50)
print("SPLITS SAVED SUCCESSFULLY")
print("=" * 50)

print(f"Train CSV: {PROCESSED_PATH / 'train.csv'}")
print(f"Val CSV:   {PROCESSED_PATH / 'val.csv'}")
print(f"Test CSV:  {PROCESSED_PATH / 'test.csv'}")