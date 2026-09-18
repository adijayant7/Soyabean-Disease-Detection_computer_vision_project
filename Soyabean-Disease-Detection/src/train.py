import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import create_model
from dataloader import train_loader, val_loader


# ==========================================
# SETTINGS
# ==========================================

NUM_EPOCHS = 20
LEARNING_RATE = 0.0001

# Automatically find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder where trained models will be saved
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# MODEL
# ==========================================

model = create_model().to(device)


# ==========================================
# LOSS FUNCTION AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAIN FOR ONE EPOCH
# ==========================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Calculate metrics
        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = (
        correct_predictions / total_predictions
    ) * 100

    return epoch_loss, epoch_accuracy


# ==========================================
# VALIDATION
# ==========================================

def validate():

    # Evaluation mode
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # No gradient calculation during validation
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_predictions += labels.size(0)

    val_loss = running_loss / len(val_loader)

    val_accuracy = (
        correct_predictions / total_predictions
    ) * 100

    return val_loss, val_accuracy


# ==========================================
# FULL TRAINING LOOP
# ==========================================

best_val_accuracy = 0.0

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    # Train
    train_loss, train_accuracy = train_one_epoch()

    # Validate
    val_loss, val_accuracy = validate()

    # Print results
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.2f}%")

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model_path = MODEL_DIR / "best_model.pth"

        torch.save(
            model.state_dict(),
            model_path
        )

        print("✓ Best model saved!")


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Model saved at: {MODEL_DIR / 'best_model.pth'}")