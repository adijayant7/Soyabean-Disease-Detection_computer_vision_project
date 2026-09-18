import torch
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import create_model
from dataloader import test_loader, test_dataset


# ==========================================
# SETTINGS
# ==========================================

CLASS_NAMES = [
    "Bacterial Blight",
    "Cercospora Leaf Blight",
    "Healthy",
    "Rust",
    "Sudden Death Syndrome"
]


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.pth"
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DETAILED MODEL EVALUATION")
print("=" * 60)

print(f"Using device: {device}")


# ==========================================
# LOAD MODEL
# ==========================================

model = create_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ==========================================
# STORE RESULTS
# ==========================================

all_predictions = []
all_labels = []
all_confidences = []


# ==========================================
# TEST ALL IMAGES
# ==========================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        # Convert raw scores into probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Get prediction and confidence
        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        # Move results back to CPU
        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_confidences.extend(
            confidences.cpu().numpy()
        )


# ==========================================
# OVERALL ACCURACY
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual class")
print("Columns = Predicted class\n")

print(cm)


# ==========================================
# MISCLASSIFIED IMAGES
# ==========================================

print("\n" + "=" * 60)
print("MISCLASSIFIED IMAGES")
print("=" * 60)

incorrect_count = 0

for index, (
    actual,
    predicted,
    confidence
) in enumerate(
    zip(
        all_labels,
        all_predictions,
        all_confidences
    )
):

    if actual != predicted:

        incorrect_count += 1

        # Get exact image path
        image_path = test_dataset.image_paths[index]

        print(f"\nImage number: {index + 1}")
        print(f"Path: {image_path}")

        print(
            f"Actual: "
            f"{CLASS_NAMES[actual]}"
        )

        print(
            f"Predicted: "
            f"{CLASS_NAMES[predicted]}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )


print(
    f"\nTotal incorrect predictions: "
    f"{incorrect_count}"
)