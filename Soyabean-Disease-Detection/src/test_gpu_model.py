import torch

from model import create_model
from test_dataloader import train_loader


# Check device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("GPU MODEL TEST")
print("=" * 50)

print(f"Device being used: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Create model and move it to GPU
model = create_model()
model = model.to(device)


# Get one real batch from DataLoader
images, labels = next(iter(train_loader))

print("\nBefore moving to GPU:")
print(f"Images device: {images.device}")
print(f"Labels device: {labels.device}")


# Move batch to GPU
images = images.to(device)
labels = labels.to(device)

print("\nAfter moving to GPU:")
print(f"Images device: {images.device}")
print(f"Labels device: {labels.device}")


# Forward pass
model.eval()

with torch.no_grad():
    outputs = model(images)


# Get predictions
predictions = torch.argmax(outputs, dim=1)


print("\n" + "=" * 50)
print("MODEL OUTPUT")
print("=" * 50)

print(f"Output shape: {outputs.shape}")
print(f"Predictions: {predictions}")
print(f"Actual labels: {labels}")