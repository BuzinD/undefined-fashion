"""Two-layer MLP with input normalization, SGD (lr=0.3), and cosine annealing."""

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 0.3


class Normalize(nn.Module):
    """Normalize input images with dataset mean and standard deviation."""

    def __init__(self, mean: float = 0.2860, std: float = 0.3530) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


def train(train_set: Dataset) -> nn.Module:
    cfg = Config()
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

    # Preload into memory once so each epoch does not re-decode images.
    all_images, all_labels = next(iter(DataLoader(train_set, batch_size=len(train_set))))
    cached_set = TensorDataset(all_images, all_labels)

    train_split, valid_split = random_split(cached_set, [50_000, 10_000])
    train_loader = DataLoader(train_split, cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_split, cfg.batch_size, shuffle=False)

    model = nn.Sequential(
        Normalize(mean=0.2860, std=0.3530),
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10),
    )
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        train_loss /= len(train_split)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                valid_loss += criterion(model(images), targets).item() * len(targets)
        valid_loss /= len(valid_split)

        scheduler.step()
        print(f"{epoch=} {train_loss=:.4f} {valid_loss=:.4f} lr={scheduler.get_last_lr()[0]:.5f}")

    return model
