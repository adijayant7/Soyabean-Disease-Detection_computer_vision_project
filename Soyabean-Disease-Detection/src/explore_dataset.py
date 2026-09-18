from pathlib import Path
from collections import Counter

DATASET_PATH=Path(r".\data\raw\soyabean_dataset")

IMAGE_EXTENTION={".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

if not DATASET_PATH.exists():
    print(f"dataset is not found at:{DATASET_PATH}")
    exit()
classes=sorted([folder for folder in DATASET_PATH.iterdir() if folder.is_dir()])

print("="*50)
print("soyabean data analysis")
print("="*50)

total_images=0
all_extention=[]

for folder in classes:
    images=[
        file for file in folder.rglob("*") if file.suffix in IMAGE_EXTENTION 
    ]
    image_count=len(images)
    total_images+=image_count

    print(f"class folder {folder}:{image_count} iamges")
    for image in images:
        all_extention.append(image.suffix.lower())
    
print("\n" + "=" * 50)
print("image formats")

extention_count=Counter(all_extention)

for extention,count in extention_count.items():
    print(f"{extention}:{count}")

print("="*50)