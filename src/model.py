"""CNN model definition and training utilities for Fashion MNIST."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    PYTORCH_MODEL_PATH,
)


class FashionCNN(nn.Module):
    """A small CNN suitable for Fashion-MNIST (28×28 grayscale images)."""

    def __init__(self) -> None:
        super().__init__()
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Data helpers ───────────────────────────────────────────────────────────────

_transform: transforms.Compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def get_datasets() -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Return (train_dataset, test_dataset)."""
    train_ds: datasets.FashionMNIST = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=_transform
    )
    test_ds: datasets.FashionMNIST = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=_transform
    )
    return train_ds, test_ds


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader)."""
    train_ds, test_ds = get_datasets()
    train_loader: DataLoader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader: DataLoader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# ── Training ───────────────────────────────────────────────────────────────────


def train_model(
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    save_path: Optional[Path] = None,
    force: bool = False,
) -> FashionCNN:
    """Train the CNN on Fashion-MNIST and save weights.

    If weights already exist at *save_path* and *force* is ``False``,
    training is skipped and the existing model is loaded instead.

    Returns the trained model (on CPU, in eval mode).
    """
    save_path = save_path or PYTORCH_MODEL_PATH
    if not force and save_path.exists():
        print(f"[train] Trained model already exists at {save_path} — skipping training.")
        print("[train] Use --force-train to retrain from scratch.")
        return load_model(save_path)

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[train] Using device: {device}")

    model: FashionCNN = FashionCNN().to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=lr)

    train_loader, _ = get_dataloaders()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        t0: float = time.time()

        for images, labels in train_loader:
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)

            optimizer.zero_grad()
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += int(predicted.eq(labels).sum().item())

        epoch_loss: float = running_loss / total
        epoch_acc: float = 100.0 * correct / total
        elapsed: float = time.time() - t0
        print(
            f"  Epoch {epoch:>2}/{epochs}  —  "
            f"loss: {epoch_loss:.4f}  acc: {epoch_acc:.2f}%  "
            f"({elapsed:.1f}s)"
        )

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    torch.save(model.state_dict(), save_path)
    print(f"[train] Model saved → {save_path}")
    return model


def load_model(path: Optional[Path] = None) -> FashionCNN:
    """Load a pre-trained FashionCNN from disk."""
    path = path or PYTORCH_MODEL_PATH
    model: FashionCNN = FashionCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
