from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


# ==========================================
# TRANSFORMATIONS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])


val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# ==========================================
# CUSTOM DATASET
# ==========================================

class SoybeanDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.image_paths = []
        self.labels = []
        self.transform = transform

        with open(
            csv_file,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                self.image_paths.append(
                    Path(row["image_path"])
                )

                self.labels.append(
                    int(row["label"])
                )


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# ==========================================
# CREATE DATASETS
# ==========================================

train_dataset = SoybeanDataset(
    SPLIT_DIR / "train.csv",
    transform=train_transform
)

val_dataset = SoybeanDataset(
    SPLIT_DIR / "validation.csv",
    transform=val_test_transform
)

test_dataset = SoybeanDataset(
    SPLIT_DIR / "test.csv",
    transform=val_test_transform
)


# ==========================================
# CREATE DATALOADERS
# ==========================================

BATCH_SIZE = 16


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("CLEAN DATALOADER TEST")
    print("=" * 50)

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images:       {len(test_dataset)}")

    images, labels = next(iter(train_loader))

    print("\nFIRST TRAINING BATCH")
    print(f"Image batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")