"""MLP from fashion.ipynb, wrapped in the leaderboard's train interface."""

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


@dataclass
class Config:
    batch_size: int = 512
    epochs: int = 40


def train(train_set: Dataset) -> nn.Module:
    cfg = Config()
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

    # Convert images to tensors once instead of repeating it on every epoch.
    images, labels = next(iter(DataLoader(train_set, batch_size=len(train_set))))
    cached_set = TensorDataset(images, labels)
    train_split, valid_split = random_split(
        cached_set,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(0),
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    model.to(device)
    train_loader = DataLoader(train_split, cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_split, cfg.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader)

        with torch.no_grad():
            valid_loss = 0.0
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)

        print(f"{epoch=} {train_loss=:.4f} {valid_loss}:.4f")

    model.eval()
    return model
