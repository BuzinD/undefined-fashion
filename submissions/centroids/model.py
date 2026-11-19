"""Nearest centroid: no gradient descent, no learned parameters.

For each class, average all its training images into one "centroid" image:
the typical sneaker, the typical bag, and so on. To label a new image, find
the centroid it is closest to.

This is a good first model to compare against. It trains in seconds and has
0 parameters, because the centroids are stored with register_buffer.
It scores below the baseline, because one average image cannot capture every
way a shirt can look.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class NearestCentroid(nn.Module):
    def __init__(self, centroids: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("centroids", centroids)  # shape (10, 784)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = images.flatten(start_dim=1)  # shape (batch, 784)
        distances = torch.cdist(pixels, self.centroids)  # shape (batch, 10)
        # The closest centroid should get the highest score, so flip the sign.
        return -distances


def train(train_set: Dataset) -> nn.Module:
    # Load all 60,000 images as one batch.
    images, labels = next(iter(DataLoader(train_set, batch_size=len(train_set))))
    pixels = images.flatten(start_dim=1)  # shape (60000, 784)

    centroids = torch.stack([pixels[labels == label].mean(dim=0) for label in range(10)])
    return NearestCentroid(centroids)
