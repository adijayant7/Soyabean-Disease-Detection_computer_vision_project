from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH=Path(r"C:\Users\sasmi\Desktop\folders\Soyabean_Disease_Detection\data\raw\soyabean_dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

if not DATASET_PATH.exists():
    print("the dataset is not found")
    exit()


classes=sorted([folder for folder in DATASET_PATH.iterdir() if folder.is_dir()])

fig,axes=plt.subplots(1,len(classes),figsize=(20,5))

for ax,class_folder in zip(axes,classes):
    images=[file for file in class_folder.rglob("*")
            if (file.is_file() and file.suffix in IMAGE_EXTENSIONS)]
    selected_image=random.choice(images)

    image=Image.open(selected_image)

    ax.imshow(image)
    ax.set_title(class_folder.name)

plt.tight_layout()
ax.axis("off")
plt.show()
