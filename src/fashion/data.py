"""Load the Fashion-MNIST images.

Every image is a 1x28x28 float tensor with values from 0 to 1.
Every label is a number from 0 to 9 (see CLASSES).
"""

from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import v2

DATA_DIR = Path("data")

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def train_set(root: Path = DATA_DIR) -> datasets.FashionMNIST:
    """The 60,000 training images. Students may use these however they like."""
    return datasets.FashionMNIST(root, train=True, download=True, transform=TRANSFORM)


def test_set(root: Path = DATA_DIR) -> datasets.FashionMNIST:
    """The 10,000 test images. Only the scoring script uses these."""
    return datasets.FashionMNIST(root, train=False, download=True, transform=TRANSFORM)
