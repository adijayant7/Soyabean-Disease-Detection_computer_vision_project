import torch
from pathlib import Path

from model import create_model
from dataloader import test_loader


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MODEL TESTING")
print("=" * 60)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# LOAD MODEL
# ==========================================

model = create_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

print(f"\nModel loaded from:")
print(MODEL_PATH)


# ==========================================
# TEST MODEL
# ==========================================

correct_predictions = 0
total_predictions = 0


with torch.no_grad():

    for images, labels in test_loader:

        # Move data to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # Get model predictions
        outputs = model(images)

        # Class with highest score
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Count correct predictions
        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_predictions += labels.size(0)


# ==========================================
# FINAL RESULTS
# ==========================================

test_accuracy = (
    correct_predictions / total_predictions
) * 100

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Total test images: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)