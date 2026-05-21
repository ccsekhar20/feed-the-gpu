import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_data(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    val_split: int = 5000,
):
    os.makedirs(data_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    train_data_augmented = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    train_data_clean = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_data = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=eval_transform,
    )

    train_size = len(train_data_augmented) - val_split

    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(
        len(train_data_augmented),
        generator=generator
    ).tolist()

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_data = Subset(train_data_augmented, train_indices)
    val_data = Subset(train_data_clean, val_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader