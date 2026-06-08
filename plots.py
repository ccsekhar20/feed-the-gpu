"""
plots.py
Generates all 5 project figures from results CSVs.
Saves PNGs to plots/ directory.

Usage:
    python plots.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = "./results"
PLOTS_DIR   = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

STYLE = {
    "figure.figsize":   (7, 4.5),
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
}
plt.rcParams.update(STYLE)
COLOR = "#4C72B0"
COLOR2 = "#DD8452"


def load_exp(df, name):
    return df[df["experiment"] == name].copy()


def save(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 1: Epoch Time vs num_workers ─────────────────────────────────────
def plot_epoch_time_vs_workers(df):
    d = load_exp(df, "workers")
    fig, ax = plt.subplots()
    ax.plot(d["num_workers"], d["epoch_time_s"], marker="o", color=COLOR, lw=2)
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Epoch Time (s)")
    ax.set_title("Epoch Time vs. Number of DataLoader Workers")
    ax.xaxis.set_major_locator(mticker.FixedLocator(d["num_workers"].tolist()))
    fig.tight_layout()
    save(fig, "epoch_time_vs_workers.png")


# ── Figure 2: Images/sec vs num_workers ─────────────────────────────────────
def plot_throughput_vs_workers(df):
    d = load_exp(df, "workers")
    fig, ax = plt.subplots()
    ax.plot(d["num_workers"], d["images_per_sec"], marker="o", color=COLOR, lw=2)
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Images / Second")
    ax.set_title("Training Throughput vs. Number of DataLoader Workers")
    ax.xaxis.set_major_locator(mticker.FixedLocator(d["num_workers"].tolist()))
    fig.tight_layout()
    save(fig, "throughput_vs_workers.png")


# ── Figure 3: Batch Size vs Throughput ──────────────────────────────────────
def plot_throughput_vs_batch(df):
    d = load_exp(df, "batch")
    fig, ax = plt.subplots()
    ax.plot(d["batch_size"], d["images_per_sec"], marker="s", color=COLOR2, lw=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Images / Second")
    ax.set_title("Training Throughput vs. Batch Size")
    ax.xaxis.set_major_locator(mticker.FixedLocator(d["batch_size"].tolist()))
    fig.tight_layout()
    save(fig, "throughput_vs_batch_size.png")


# ── Figure 4: Preprocessing Delay vs Throughput ─────────────────────────────
def plot_throughput_vs_delay(df):
    d = load_exp(df, "delay")
    fig, ax = plt.subplots()
    ax.plot(d["delay_ms"], d["images_per_sec"], marker="^", color="crimson", lw=2)
    ax.set_xlabel("Artificial Preprocessing Delay (ms)")
    ax.set_ylabel("Images / Second")
    ax.set_title("Throughput Drops as Preprocessing Delay Increases")
    ax.xaxis.set_major_locator(mticker.FixedLocator(d["delay_ms"].tolist()))
    fig.tight_layout()
    save(fig, "throughput_vs_delay.png")


# ── Figure 5: Validation Accuracy Across All Experiments ────────────────────
def plot_val_accuracy(df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    exp_info = [
        ("workers", "num_workers",  "num_workers",  COLOR),
        ("pin",     "pin_memory",   "pin_memory",   COLOR2),
        ("batch",   "batch_size",   "batch_size",   "seagreen"),
        ("delay",   "delay_ms",     "delay (ms)",   "crimson"),
    ]

    for ax, (exp_name, x_col, x_label, color) in zip(axes, exp_info):
        d = load_exp(df, exp_name)
        ax.bar(d[x_col].astype(str), d["val_acc"] * 100, color=color, alpha=0.8)
        ax.set_xlabel(x_label)
        ax.set_title(exp_name.capitalize())
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    axes[0].set_ylabel("Validation Accuracy (%)")
    fig.suptitle("Validation Accuracy Across Experiment Settings", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "val_accuracy_across_experiments.png")


# ── Bonus: Batch Load Time vs Step Time (stacked bar) ───────────────────────
def plot_time_breakdown_vs_workers(df):
    d = load_exp(df, "workers")
    fig, ax = plt.subplots()
    ax.bar(d["num_workers"].astype(str), d["avg_batch_load_ms"],
           label="Batch Load Time", color=COLOR2, alpha=0.85)
    ax.bar(d["num_workers"].astype(str), d["avg_step_ms"],
           bottom=d["avg_batch_load_ms"], label="Training Step Time",
           color=COLOR, alpha=0.85)
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Data Load vs. Training Step Time per Batch")
    ax.legend()
    fig.tight_layout()
    save(fig, "time_breakdown_vs_workers.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    dl_path  = os.path.join(RESULTS_DIR, "dataloader_results.csv")
    base_path = os.path.join(RESULTS_DIR, "baseline_results.csv")

    if not os.path.exists(dl_path):
        print(f"ERROR: {dl_path} not found. Run experiments.py first.")
        return

    df = pd.read_csv(dl_path)
    print(f"Loaded {len(df)} rows from {dl_path}\n")

    print("Generating plots...")
    plot_epoch_time_vs_workers(df)
    plot_throughput_vs_workers(df)
    plot_throughput_vs_batch(df)
    plot_throughput_vs_delay(df)
    plot_val_accuracy(df)
    plot_time_breakdown_vs_workers(df)

    if os.path.exists(base_path):
        bdf = pd.read_csv(base_path)
        fig, ax = plt.subplots()
        ax.plot(bdf["epoch"], bdf["val_acc"] * 100, marker="o", color=COLOR, lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Baseline Training Curve (CIFAR-10 CNN)")
        fig.tight_layout()
        save(fig, "baseline_training_curve.png")

    print("\nDone! All plots saved to ./plots/")


if __name__ == "__main__":
    main()