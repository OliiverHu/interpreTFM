from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_hist(values, title: str, xlabel: str, out_path: Path, bins: int = 50, log_x: bool = False):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    x = values
    if log_x:
        x = [v for v in x if v > 0]
    plt.hist(x, bins=bins)
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(
    x_values,
    y_values,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    log_x: bool = False,
    log_y: bool = False,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    x = pd.Series(x_values, dtype="float64")
    y = pd.Series(y_values, dtype="float64")
    mask = x.notna() & y.notna()
    if log_x:
        mask &= x > 0
    if log_y:
        mask &= y > 0
    x = x[mask]
    y = y[mask]

    plt.scatter(x, y, s=8, alpha=0.35)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
