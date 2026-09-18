import torch
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ==========================================
# LOAD MODEL
# ==========================================

model = create_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()


# ==========================================
# GET PREDICTIONS
# ==========================================

all_predictions = []
all_labels = []
all_confidences = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

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
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

fig, ax = plt.subplots(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("Soybean Disease Detection - Confusion Matrix")
plt.tight_layout()

confusion_path = (
    RESULTS_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nConfusion matrix saved at:")
print(confusion_path)


# ==========================================
# FIND MISCLASSIFIED IMAGES
# ==========================================

incorrect_indices = []

for index, (actual, predicted) in enumerate(
    zip(all_labels, all_predictions)
):

    if actual != predicted:

        incorrect_indices.append(index)


print(f"\nIncorrect predictions: {len(incorrect_indices)}")


# ==========================================
# VISUALIZE MISCLASSIFIED IMAGES
# ==========================================

if incorrect_indices:

    fig, axes = plt.subplots(
        1,
        len(incorrect_indices),
        figsize=(12, 6)
    )

    # Handles the case of only one incorrect image
    if len(incorrect_indices) == 1:
        axes = [axes]


    for ax, index in zip(
        axes,
        incorrect_indices
    ):

        image_path = (
            test_dataset.image_paths[index]
        )

        image = plt.imread(image_path)

        actual = all_labels[index]
        predicted = all_predictions[index]
        confidence = all_confidences[index]

        ax.imshow(image)

        ax.set_title(
            f"Actual: {CLASS_NAMES[actual]}\n"
            f"Predicted: {CLASS_NAMES[predicted]}\n"
            f"Confidence: {confidence * 100:.2f}%"
        )

        ax.axis("off")


    plt.tight_layout()

    incorrect_path = (
        RESULTS_DIR / "misclassified_images.png"
    )

    plt.savefig(
        incorrect_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nMisclassified images saved at:")
    print(incorrect_path)


print("\nVISUALIZATION COMPLETED!")