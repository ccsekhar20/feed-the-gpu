import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import load_data
from model import CifarCNN


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        predicted_labels = predictions.argmax(dim=1)
        total_correct += predicted_labels.eq(labels).sum().item()
        total_images += batch_size

    avg_loss = total_loss / total_images
    accuracy = 100.0 * total_correct / total_images

    return avg_loss, accuracy


def evaluate(model, val_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            predicted_labels = predictions.argmax(dim=1)
            total_correct += predicted_labels.eq(labels).sum().item()
            total_images += batch_size

    avg_loss = total_loss / total_images
    accuracy = 100.0 * total_correct / total_images

    return avg_loss, accuracy


def save_results(results_path, rows):
    with open(results_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "epoch_time_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Baseline CIFAR-10 training run")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    train_loader, val_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = CifarCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.75,
    )

    results = []
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        results.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 2),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 2),
            "epoch_time_seconds": round(epoch_time, 2),
        })

    total_time = time.time() - total_start_time

    results_path = os.path.join(args.results_dir, "baseline_results.csv")
    model_path = os.path.join(args.results_dir, "baseline_model.pth")

    save_results(results_path, results)
    torch.save(model.state_dict(), model_path)

    print("Training complete.")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Saved results to {results_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()