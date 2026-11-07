"""Training curves of the five diagnostic runs of question Q3 (journal section 16.3).

    python plots_q3.py --out fig_ta_q3.png

Left panel: training loss, which is what the question is about (does the GATv2 model underfit).
Right panel: validation loss, shown because it ranks the runs differently; earlier sections have
established that validation MSE does not predict the coverage cost for these models.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# fixed order, one hue and one marker per run; references are drawn thicker and labelled first
SERIES = [("V2", "V2 as trained before (reference)", "#eb6834", "s", "-", 2.6),
          ("V1", "V1, no map, TAGConv K=3 (reference)", "#184f95", "o", "-", 2.6),
          ("V2lr", "V2, learning rate x10", "#f0a07a", "v", "--", 1.8),
          ("V2res", "V2 + residual, LayerNorm, bounded edges", "#8d4bbb", "D", "-", 1.8),
          ("V2deep", "same, 15 layers", "#1baf7a", "^", "--", 1.8),
          ("V2logdeg", "same + log-degree feature", "#e87ba4", "P", "--", 1.8),
          ("V1K1", "V1 with TAGConv K=1 (5 hops)", "#6da7ec", "X", "--", 1.8)]


def panel(ax: plt.Axes, frame: pd.DataFrame, column: str, title: str) -> None:
    for run, label, colour, marker, style, width in SERIES:
        rows = frame[frame.run == run].sort_values("epoch")
        ax.plot(rows.epoch, rows[column], color=colour, marker=marker, markersize=4.5, markevery=3,
                linestyle=style, linewidth=width, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error on the control")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", default="results_Q3/q3_losses.csv")
    parser.add_argument("--out", default="fig_ta_q3.png")
    args = parser.parse_args()
    frame = pd.read_csv(args.losses)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=150, sharex=True)
    panel(axes[0], frame, "train_loss", "Q3. Training loss: the plateau of V2 is an optimization\nproblem, not a lack of capacity")
    panel(axes[1], frame, "val_loss", "Validation loss (does not predict the coverage cost,\nsee section 14)")
    axes[0].legend(fontsize=8.5, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
