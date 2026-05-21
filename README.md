# feed-the-gpu

This repository is a reproducible GPU training workload for CIFAR-10. The goal is to provide a stable, repeatable training pipeline so the project can later be used for dataloader and throughput benchmarking.

## Setup

1. Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run training

```bash
python train.py --epochs 5
```

The first run will download CIFAR-10 and save baseline artifacts in the `results/` directory.

## Project structure

- `train.py` — main training script
- `model.py` — baseline CNN architecture
- `data.py` — CIFAR-10 download, normalization, and DataLoader creation
- `results/` — saved model state and baseline results
- `data/` — CIFAR-10 dataset cache

## Notes

- The focus is on building a reproducible training workload rather than optimizing accuracy.
- `train.py` prints training loss, validation accuracy, and runtime per epoch.
- Baseline results and the model checkpoint are saved under `results/`.
