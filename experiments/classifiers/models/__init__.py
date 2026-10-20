"""Classifier architectures used in the paper (Table S1)."""

from .mnist_cnn import MnistCNN
from .mobilenetv2 import MobileNetV2
from .preact_resnet import PreActResNet18
from .regnet import RegNetX_200MF
from .resnet import ResNet18
from .vgg import VGG

# The five CIFAR-10 architectures (two seeds each in the paper).
CIFAR10_ARCHITECTURES = {
    "VGG": lambda: VGG("VGG16"),
    "ResNet": ResNet18,
    "PreActResNet": PreActResNet18,
    "MobileNet": MobileNetV2,
    "RegNetX": RegNetX_200MF,
}


def build_cifar10_model(name):
    if name not in CIFAR10_ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}; choose from {list(CIFAR10_ARCHITECTURES)}")
    return CIFAR10_ARCHITECTURES[name]()
