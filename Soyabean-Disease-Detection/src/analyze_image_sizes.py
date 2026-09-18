from pathlib import Path
from PIL import Image
from collections import Counter

# Dataset location
DATASET_PATH = Path(r"C:\Users\sasmi\Desktop\folders\Soyabean_Disease_Detection\data\raw\soyabean_dataset")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Check if dataset exists
if not DATASET_PATH.exists():
    print("Dataset not found")
    exit()

# Get all image files from the dataset
images = [
    file
    for file in DATASET_PATH.rglob("*")
    if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total images found: {len(images)}")


widths=[]
heights=[]
sizes=[]

for image_path in images:
    try:
        with Image.open(image_path) as image:
            width, height = image.size

            widths.append(width)
            heights.append(height)
            sizes.append((width, height))

            
    except Exception as ex:
        print(f"there is some error: {ex}")
    # Print results
print("\n" + "=" * 50)
print("IMAGE SIZE ANALYSIS")
print("=" * 50)

print(f"Minimum width: {min(widths)}")
print(f"Maximum width: {max(widths)}")
print(f"Average width: {sum(widths) / len(widths):.2f}")

print()

print(f"Minimum height: {min(heights)}")
print(f"Maximum height: {max(heights)}")
print(f"Average height: {sum(heights) / len(heights):.2f}")

# Find common image dimensions
size_counts = Counter(sizes)

print("\nMost common image sizes:")

for size, count in size_counts.most_common(10):
    print(f"{size}: {count} images")