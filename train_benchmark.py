# ============================================================
# train_benchmark.py
# Project: When does data loading bottleneck GPU training?
# ============================================================
# HOW TO USE THIS FILE:
#   - Each section is labeled with who owns it
#   - Fill in the TODOs in your section
#   - Don't touch other people's sections until combining
# ============================================================


# ---- EVERYONE: run this first in Colab to install packages ----
# !pip install torch torchvision pynvml


import time
import csv
import os
from dataclasses import dataclass, fields, asdict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Optional: GPU utilization tracking
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


# ============================================================
# SECTION 1 — PERSON 1: Model
# ============================================================

class ResidualBlock(nn.Module):
    # TODO: define a basic residual block
    # - two 3x3 conv layers
    # - batch norm after each conv
    # - ReLU activation
    # - skip connection (add input to output)
    def __init__(self, channels):
        super().__init__()
        # TODO: define self.block and self.relu
        pass

    def forward(self, x):
        # TODO: return relu(x + self.block(x))
        pass


class SmallResNet(nn.Module):
    # TODO: define a small CNN for CIFAR-10 (32x32 images, 10 classes)
    # Suggested structure:
    #   stem:   Conv2d(3 -> 32) + BN + ReLU
    #   layer1: two ResidualBlocks at 32 channels
    #   layer2: downsample to 64 channels + one ResidualBlock
    #   layer3: downsample to 128 channels + one ResidualBlock
    #   head:   AdaptiveAvgPool -> Flatten -> Linear(128 -> 10)
    def __init__(self, num_classes=10):
        super().__init__()
        # TODO: define self.stem, self.layer1, self.layer2, self.layer3, self.head
        pass

    def forward(self, x):
        # TODO: pass x through each layer in order
        pass


# ============================================================
# SECTION 1 — PERSON 1: Training loop
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, n_train):
    """
    Run one full pass over the training data.
    Returns: (epoch_time_seconds, images_per_second, avg_gpu_utilization)
    """
    model.train()
    gpu_samples = []

    # TODO: start a timer here using time.perf_counter()
    t0 = None

    for imgs, labels in loader:
        # TODO: move imgs and labels to device (use non_blocking=True)

        # TODO: zero the gradients (use set_to_none=True)

        # TODO: forward pass through model

        # TODO: compute loss using criterion

        # TODO: backward pass

        # TODO: optimizer step

        # collect GPU utilization each batch
        util = get_gpu_utilization()
        if util is not None:
            gpu_samples.append(util)

    # TODO: stop the timer and compute:
    #   elapsed = time since t0
    #   imgs_per_sec = n_train / elapsed
    #   gpu_util = average of gpu_samples (or -1.0 if empty)
    elapsed = None
    imgs_per_sec = None
    gpu_util = None

    return elapsed, imgs_per_sec, gpu_util


def evaluate(model, loader, device):
    """Returns validation accuracy as a float between 0 and 1."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            # TODO: move to device, get predictions, count correct
            pass
    return correct / total


# ============================================================
# SECTION 2 — PERSON 2: Dataset setup
# ============================================================

class SlowCIFAR10(torchvision.datasets.CIFAR10):
    """
    CIFAR-10 with an optional artificial delay per sample.
    This simulates a slow preprocessing pipeline.
    """
    def __init__(self, *args, delay_seconds=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_seconds = delay_seconds

    def __getitem__(self, index):
        # TODO: if delay_seconds > 0, sleep for that long before returning
        # then call super().__getitem__(index) to get the actual sample
        pass


def get_cifar10(root="./data", delay_seconds=0.0):
    """
    Download CIFAR-10 and return (train_dataset, val_dataset).

    Transforms to apply:
      Train: RandomCrop(32, padding=4), RandomHorizontalFlip,
             ToTensor, Normalize(mean, std)
      Val:   ToTensor, Normalize(mean, std)

    CIFAR-10 mean: (0.4914, 0.4822, 0.4465)
    CIFAR-10 std:  (0.2470, 0.2435, 0.2616)
    """
    # TODO: define train_transform using transforms.Compose
    train_transform = None

    # TODO: define val_transform (no random augmentation, just ToTensor + Normalize)
    val_transform = None

    # TODO: create train_ds using SlowCIFAR10
    #   root=root, train=True, download=True, transform=train_transform, delay_seconds=delay_seconds
    train_ds = None

    # TODO: create val_ds using SlowCIFAR10
    #   root=root, train=False, download=True, transform=val_transform, delay_seconds=0.0
    #   (no delay on validation — we're only benchmarking training)
    val_ds = None

    return train_ds, val_ds


# ============================================================
# SECTION 2 — PERSON 2: Experiment configurations
# ============================================================

def build_experiment_grid():
    """
    Returns a list of configs to benchmark, each as a tuple:
        (num_workers, pin_memory, batch_size, delay_seconds)

    Experiment 1 — vary num_workers (how many parallel data loading processes)
        workers: 0, 1, 2, 4, 8
        keep everything else fixed: pin_memory=True, batch_size=128, delay=0.0

    Experiment 2 — pinned vs normal memory
        pin_memory: False, True
        keep everything else fixed: num_workers=4, batch_size=128, delay=0.0

    Experiment 3 — vary batch size
        batch_size: 32, 64, 128, 256
        keep everything else fixed: num_workers=4, pin_memory=True, delay=0.0

    Experiment 4 — artificial delay (simulates slow preprocessing)
        delay: 0.0, 0.0005, 0.001, 0.005
        keep everything else fixed: num_workers=4, pin_memory=True, batch_size=128
    """
    configs = []

    # TODO: add tuples for Experiment 1 (vary num_workers)

    # TODO: add tuples for Experiment 2 (pin_memory)

    # TODO: add tuples for Experiment 3 (batch_size)

    # TODO: add tuples for Experiment 4 (delay)

    # remove duplicates while keeping order
    seen = set()
    unique = []
    for c in configs:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ============================================================
# SECTION 2 — PERSON 2: DataLoader setup (inside run_experiment)
# ============================================================

def make_dataloaders(train_ds, val_ds, batch_size, num_workers, pin_memory):
    """
    Create and return (train_loader, val_loader).

    train_loader settings to use:
      - batch_size = batch_size (from experiment config)
      - shuffle = True
      - num_workers = num_workers (from experiment config)
      - pin_memory = pin_memory (from experiment config)
      - persistent_workers = True if num_workers > 0 else False

    val_loader settings (fixed — not part of the experiment):
      - batch_size = 256
      - shuffle = False
      - num_workers = 2
      - pin_memory = pin_memory
    """
    # TODO: create train_loader using DataLoader
    train_loader = None

    # TODO: create val_loader using DataLoader
    val_loader = None

    return train_loader, val_loader


# ============================================================
# SECTION 3 — PERSON 3: GPU utility + result logging
# ============================================================

def get_gpu_utilization(device_index=0):
    """
    Return current GPU utilization as a percentage (0-100).
    Returns None if pynvml is not available.
    """
    if not NVML_AVAILABLE:
        return None
    # TODO: use pynvml to get GPU utilization
    #   pynvml.nvmlDeviceGetHandleByIndex(device_index)
    #   pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    pass


@dataclass
class RunResult:
    # TODO: add fields for everything you want to log per epoch:
    #   config fields: num_workers, pin_memory, batch_size, delay_seconds
    #   measurement fields: epoch, epoch_time_s, imgs_per_sec, gpu_util_pct, val_accuracy
    pass


def save_csv(results, path="results/results.csv"):
    """Save a list of RunResult objects to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        print("No results to save.")
        return
    # TODO: open the CSV file and write:
    #   - a header row using the field names from RunResult
    #   - one row per result using asdict()
    print(f"Results saved to {path}")


# ============================================================
# SECTION 3 — PERSON 3: Experiment runner + main
# ============================================================

def run_experiment(num_workers, pin_memory, batch_size, delay_seconds,
                   num_epochs, device, results):
    """Run one config for num_epochs and append RunResult objects to results."""
    print(f"\nRunning: workers={num_workers} | pin={pin_memory} | "
          f"bs={batch_size} | delay={delay_seconds}")

    train_ds, val_ds = get_cifar10(delay_seconds=delay_seconds)
    train_loader, val_loader = make_dataloaders(
        train_ds, val_ds, batch_size, num_workers, pin_memory
    )

    model     = SmallResNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        epoch_t, ips, gpu_util = train_one_epoch(
            model, train_loader, optimizer, criterion, device, len(train_ds)
        )
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        # TODO: create a RunResult and append to results
        #   round epoch_t to 3 decimal places
        #   round ips to 1 decimal place
        #   round gpu_util to 1 decimal place
        #   round val_acc * 100 to 2 decimal places

        print(f"  Epoch {epoch} | time={epoch_t:.1f}s | "
              f"ips={ips:.0f} | gpu={gpu_util:.0f}% | acc={val_acc*100:.1f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    configs   = build_experiment_grid()
    results   = []
    NUM_EPOCHS = 5   # increase for final runs, use 2 for quick testing

    for (workers, pin, bs, delay) in configs:
        run_experiment(
            num_workers=workers,
            pin_memory=pin,
            batch_size=bs,
            delay_seconds=delay,
            num_epochs=NUM_EPOCHS,
            device=device,
            results=results,
        )

    save_csv(results)


if __name__ == "__main__":
    main()