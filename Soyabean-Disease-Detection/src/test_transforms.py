from pathlib import Path
import random

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


DATASET_PATH = Path(
    r"C:\Users\sasmi\Desktop\folders\Soyabean_Disease_Detection\data\raw\soyabean_dataset"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# Training transformations
train_transforms = transforms.Compose([

    transforms.RandomResizedCrop(
        size=224,
        scale=(0.7, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    )
])


# Get all images
images = [
    file
    for file in DATASET_PATH.rglob("*")
    if file.is_file()
    and file.suffix.lower() in IMAGE_EXTENSIONS
]


# Select one random image
selected_image = random.choice(images)

# Open original image
original_image = Image.open(selected_image).convert("RGB")


# Create figure
fig, axes = plt.subplots(1, 5, figsize=(20, 5))


# Show original image
axes[0].imshow(original_image)
axes[0].set_title("Original")
axes[0].axis("off")


# Apply augmentation 4 times
for i in range(1, 5):

    augmented_image = train_transforms(original_image)

    axes[i].imshow(augmented_image)
    axes[i].set_title(f"Augmented {i}")
    axes[i].axis("off")


plt.tight_layout()
plt.show()