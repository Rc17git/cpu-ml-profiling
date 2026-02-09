import torch
import torchvision.models as models


def get_resnet18(num_classes: int = 10):
    """
    Returns a ResNet-18 model for CPU inference benchmarking.
    """
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
