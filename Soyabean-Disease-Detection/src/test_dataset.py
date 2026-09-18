from pathlib import Path

from dataset import SoybeanDataset
from preprocessing import train_transforms, val_test_transforms


# Find project root automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"


# Create datasets
train_dataset = SoybeanDataset(
    csv_file=TRAIN_CSV,
    transform=train_transforms
)

val_dataset = SoybeanDataset(
    csv_file=VAL_CSV,
    transform=val_test_transforms
)

test_dataset = SoybeanDataset(
    csv_file=TEST_CSV,
    transform=val_test_transforms
)


# Print dataset sizes
print("=" * 50)
print("DATASET TEST")
print("=" * 50)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")


# Test one training sample
image, label = train_dataset[0]

print("\n" + "=" * 50)
print("FIRST TRAINING SAMPLE")
print("=" * 50)

print(f"Image shape: {image.shape}")
print(f"Label: {label}")
print(f"Image data type: {image.dtype}")