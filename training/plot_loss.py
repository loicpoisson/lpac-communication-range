"""Training / validation loss curves of one or more runs of train_launch.py.

    python plot_loss.py --runs label=/path/to/models [label2=/path2] --out fig_train_loss.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="label=path pairs (path holds loss_history.json)")
    parser.add_argument("--out", default="fig_train_loss.png")
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for color, run in zip(COLORS, args.runs):
        label, path = run.split("=", 1)
        history = json.loads((Path(path) / "loss_history.json").read_text())
        epochs = range(1, len(history["train"]) + 1)
        ax.plot(epochs, history["train"], color=color, linewidth=2, label=f"{label}: training")
        ax.plot(epochs, history["val"], color=color, linewidth=2, linestyle="--", label=f"{label}: validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on normalized actions")
    ax.set_title("LPAC-K3 retraining from scratch: loss curves", fontsize=10)
    ax.grid(True, color="#dddddd")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
