"""
experiments.py
Runs all 4 DataLoader experiments and saves results to results/dataloader_results.csv

Experiments:
  1. num_workers   = 0, 1, 2, 4, 8
  2. pin_memory    = False vs True
  3. batch_size    = 32, 64, 128, 256
  4. artificial preprocessing delay = 0, 2, 5, 10 ms

Usage:
    python experiments.py              # all experiments
    python experiments.py --exp workers
    python experiments.py --exp pin
    python experiments.py --exp batch
    python experiments.py --exp delay
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import get_model

SEED = 42
torch.manual_seed(SEED)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2470, 0.2435, 0.2616]

# ── dataset with optional artificial delay ───────────────────────────────────

class DelayedCIFAR10(datasets.CIFAR10):
    """CIFAR-10 with a configurable per-sample preprocessing delay (seconds)."""
    def __init__(self, delay_s: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.delay_s = delay_s

    def __getitem__(self, index):
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        return super().__getitem__(index)


def make_loader(batch_size=64, num_workers=2, pin_memory=True,
                delay_ms=0, data_dir="./data"):
    normalize = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = DelayedCIFAR10(
        delay_s=delay_ms / 1000.0,
        root=data_dir, train=True, download=True, transform=tf,
    )
    # Use a small fixed subset (5000 samples) so experiments finish fast
    subset_size = 5000
    subset, _ = random_split(
        dataset, [subset_size, len(dataset) - subset_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    return DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        shuffle=True,
    )


# ── measure one config ───────────────────────────────────────────────────────

def measure_config(loader, device, n_epochs=2):
    """
    Train for n_epochs and return averaged metrics:
      epoch_time_s, images_per_sec, avg_batch_load_ms, avg_step_ms, val_acc
    """
    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch_times, imgs_per_secs, load_mses, step_mses = [], [], [], []
    n_samples = len(loader.dataset)

    for _ in range(n_epochs):
        model.train()
        t_epoch = time.perf_counter()
        batch_loads, steps = [], []

        t_load = time.perf_counter()
        for inputs, labels in loader:
            load_ms = (time.perf_counter() - t_load) * 1000
            batch_loads.append(load_ms)

            t_step = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            steps.append((time.perf_counter() - t_step) * 1000)

            t_load = time.perf_counter()

        epoch_time = time.perf_counter() - t_epoch
        epoch_times.append(epoch_time)
        imgs_per_secs.append(n_samples / epoch_time)
        load_mses.append(sum(batch_loads) / len(batch_loads))
        step_mses.append(sum(steps) / len(steps))

    # quick val accuracy (reuse train loader for simplicity)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
    val_acc = correct / total

    def avg(lst): return sum(lst) / len(lst)
    return {
        "epoch_time_s":      round(avg(epoch_times), 3),
        "images_per_sec":    round(avg(imgs_per_secs), 1),
        "avg_batch_load_ms": round(avg(load_mses), 3),
        "avg_step_ms":       round(avg(step_mses), 3),
        "val_acc":           round(val_acc, 4),
    }


# ── experiments ──────────────────────────────────────────────────────────────

def run_experiment(name, configs, device, writer):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}")
    for cfg in configs:
        print(f"  Running: {cfg}")
        loader = make_loader(**{k: v for k, v in cfg.items() if k != "label"})
        metrics = measure_config(loader, device)
        row = {"experiment": name, **{k: v for k, v in cfg.items() if k != "data_dir"}, **metrics}
        writer.writerow(row)
        print(f"    → {metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                        choices=["all", "workers", "pin", "batch", "delay"])
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = os.path.join(RESULTS_DIR, "dataloader_results.csv")
    fieldnames = [
        "experiment", "label",
        "batch_size", "num_workers", "pin_memory", "delay_ms",
        "epoch_time_s", "images_per_sec",
        "avg_batch_load_ms", "avg_step_ms", "val_acc",
    ]

    # Default fixed values
    D_BATCH   = 64
    D_WORKERS = 2
    D_PIN     = True
    D_DELAY   = 0

    experiments = {
        "workers": [
            {"label": f"workers={w}", "batch_size": D_BATCH, "num_workers": w,
             "pin_memory": D_PIN, "delay_ms": D_DELAY, "data_dir": args.data_dir}
            for w in [0, 1, 2, 4, 8]
        ],
        "pin": [
            {"label": f"pin={p}", "batch_size": D_BATCH, "num_workers": D_WORKERS,
             "pin_memory": p, "delay_ms": D_DELAY, "data_dir": args.data_dir}
            for p in [False, True]
        ],
        "batch": [
            {"label": f"batch={b}", "batch_size": b, "num_workers": D_WORKERS,
             "pin_memory": D_PIN, "delay_ms": D_DELAY, "data_dir": args.data_dir}
            for b in [32, 64, 128, 256]
        ],
        "delay": [
            {"label": f"delay={d}ms", "batch_size": D_BATCH, "num_workers": D_WORKERS,
             "pin_memory": D_PIN, "delay_ms": d, "data_dir": args.data_dir}
            for d in [0, 2, 5, 10]
        ],
    }

    to_run = experiments if args.exp == "all" else {args.exp: experiments[args.exp]}

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, configs in to_run.items():
            run_experiment(name, configs, device, writer)

    print(f"\nAll results saved to {csv_path}")


if __name__ == "__main__":
    main()