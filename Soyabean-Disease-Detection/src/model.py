import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


NUM_CLASSES = 5


def create_model():

    # Load pretrained EfficientNet-B0
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    # Get the number of input features
    num_features = model.classifier[1].in_features

    # Replace the original classifier
    model.classifier[1] = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return model