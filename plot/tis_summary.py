from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_combined_summary(summary_df: pd.DataFrame, out_root: Path, ensure_dir) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[warn] matplotlib unavailable, skipping plots:", e)
        return

    plot_dir = ensure_dir(out_root / "plots_cross_model")
    x = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width, summary_df["tis_mean"].astype(float), width, label="model")
    plt.bar(x, summary_df["shuffle_mean"].astype(float), width, label="shuffle")
    if "pca_mean" in summary_df.columns:
        plt.bar(x + width, summary_df["pca_mean"].astype(float), width, label="PCA")
    plt.xticks(x, summary_df["model"].astype(str).tolist())
    plt.ylabel("Mean TIS/MIS")
    plt.title("TIS validation across model activations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "tis_summary_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, summary_df["tis_mean"].astype(float) - summary_df["shuffle_mean"].astype(float))
    plt.xticks(x, summary_df["model"].astype(str).tolist())
    plt.ylabel("Mean TIS - shuffled")
    plt.title("TIS improvement over shuffled baseline")
    plt.tight_layout()
    plt.savefig(plot_dir / "tis_delta_vs_shuffle.png", dpi=200)
    plt.close()
