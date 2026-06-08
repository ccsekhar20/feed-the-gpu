"""
model.py
CNN model for CIFAR-10 classification.
Used across all experiments to ensure consistent architecture.
"""

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    Custom CNN for CIFAR-10.
    Architecture:
      - 3 convolutional blocks (Conv -> BN -> ReLU -> MaxPool)
      - 2 fully connected layers
      - 10-class output
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32x32 -> 16x16
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16x16 -> 8x8
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 8x8 -> 4x4
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x


def get_model(device: torch.device) -> CIFAR10CNN:
    """Instantiate and move model to device."""
    model = CIFAR10CNN(num_classes=10)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Sanity-check forward pass
    dummy = torch.randn(4, 3, 32, 32).to(device)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # should be (4, 10)