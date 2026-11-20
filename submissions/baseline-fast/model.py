"""Baseline with fast data handling: same model, optimizer and epochs, much less waiting.

Two changes from submissions/baseline:

1. Load all images at once from the raw pixels, instead of converting
   60,000 images one at a time.
2. Make batches by slicing one big tensor, instead of a DataLoader that
   fetches and stacks the images one by one for every batch.

The model and the training math are the same, so the accuracy is the same
(up to the random shuffle order).
"""

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    batch_size: int = 512
    epochs: int = 20
    learning_rate: float = 0.01


def load_all(train_set: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every training image and label as two tensors."""
    if hasattr(train_set, "data") and hasattr(train_set, "targets"):
        # Fashion-MNIST keeps its pixels as one uint8 tensor (60000x28x28, values 0-255).
        # Scaling it ourselves gives exactly the images the dataset would return.
        images = train_set.data.unsqueeze(1).float() / 255  # shape (60000, 1, 28, 28)
        return images, train_set.targets
    # Any other dataset: the slow way, converting one image at a time.
    return next(iter(DataLoader(train_set, batch_size=len(train_set))))


def batches(images: torch.Tensor, labels: torch.Tensor, batch_size: int):
    """Yield the images and labels in random order, batch_size at a time."""
    order = torch.randperm(len(images), device=images.device)
    for start in range(0, len(images), batch_size):
        picked = order[start : start + batch_size]
        yield images[picked], labels[picked]


def train(train_set: Dataset) -> nn.Module:
    cfg = Config()
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

    images, labels = load_all(train_set)
    images, labels = images.to(device), labels.to(device)

    # Keep 10,000 training images aside to check how well the model does on images
    # it did not train on. You never get the real test images.
    shuffled = torch.randperm(len(images), device=device)
    train_images, train_labels = images[shuffled[:50_000]], labels[shuffled[:50_000]]
    valid_images, valid_labels = images[shuffled[50_000:]], labels[shuffled[50_000:]]

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10),
    )
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()  # training mode (matters for layers like dropout)
        train_loss = 0.0
        for batch_images, batch_labels in batches(train_images, train_labels, cfg.batch_size):
            optimizer.zero_grad()
            loss = criterion(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_labels)
        train_loss /= len(train_images)

        model.eval()  # evaluation mode for checking the valid images
        with torch.no_grad():  # all 10,000 valid images fit in one batch
            valid_loss = criterion(model(valid_images), valid_labels).item()

        print(f"{epoch=} {train_loss=:.4f} {valid_loss=:.4f}")

    return model
