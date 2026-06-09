from torch import nn

from src.models.compact_cnn import compact_cnn
from src.models.resnet_cifar import resnet18_cifar


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    normalized_name = name.lower()
    if normalized_name == "resnet18_cifar":
        return resnet18_cifar(num_classes=num_classes)
    if normalized_name == "compact_cnn":
        return compact_cnn(num_classes=num_classes)
    raise ValueError(
        f"Unknown model '{name}'. Expected one of: resnet18_cifar, compact_cnn."
    )


__all__ = ["build_model", "compact_cnn", "resnet18_cifar"]
