from pathlib import Path

from torch.utils.data import DataLoader

from dataset import SoybeanDataset
from preprocessing import train_transforms, val_test_transforms


# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# CSV paths
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"


# Create datasets
train_dataset = SoybeanDataset(
    TRAIN_CSV,
    transform=train_transforms
)

val_dataset = SoybeanDataset(
    VAL_CSV,
    transform=val_test_transforms
)

test_dataset = SoybeanDataset(
    TEST_CSV,
    transform=val_test_transforms
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# Test one training batch
images, labels = next(iter(train_loader))

print("=" * 50)
print("DATALOADER TEST")
print("=" * 50)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

print("\nFIRST TRAINING BATCH")
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")
print(f"Labels: {labels}")