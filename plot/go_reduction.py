from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    fmt: str = ".2f",
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5.5))
    arr = pivot.values
    im = plt.imshow(arr, aspect="auto")
    plt.xticks(range(pivot.shape[1]), [str(x) for x in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), [str(x) for x in pivot.index])
    plt.xlabel("max_descendants")
    plt.ylabel("f1_min")
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if pd.isna(val):
                txt = "nan"
            elif fmt == "d":
                txt = str(int(round(float(val))))
            else:
                txt = format(float(val), fmt)
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)
    plt.colorbar(im, label=cbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_nmi_lines(sweep: pd.DataFrame, *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for max_desc, g in sweep.groupby("max_descendants"):
        g = g.sort_values("f1_min")
        plt.plot(g["f1_min"], g["nmi"], marker="o", label=str(max_desc))
    plt.xlabel("f1_min")
    plt.ylabel("NMI")
    plt.title(title)
    plt.legend(title="max_desc", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cross_model_best(best_df: pd.DataFrame, *, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots_cross_model"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if best_df.empty or "best_nmi" not in best_df.columns:
        return
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(best_df))
    plt.bar(x, best_df["best_nmi"].astype(float).values)
    plt.xticks(x, best_df["model"].astype(str).tolist())
    plt.ylabel("Best NMI")
    plt.title("Best GO parent NMI by model")
    plt.tight_layout()
    plt.savefig(plot_dir / "best_nmi_by_model.png", dpi=200)
    plt.close()
    if "best_n_concepts" in best_df.columns:
        plt.figure(figsize=(7, 4.5))
        plt.bar(x, best_df["best_n_concepts"].astype(float).values)
        plt.xticks(x, best_df["model"].astype(str).tolist())
        plt.ylabel("# concepts at best setting")
        plt.title("# GO concepts retained at best NMI setting")
        plt.tight_layout()
        plt.savefig(plot_dir / "best_n_concepts_by_model.png", dpi=200)
        plt.close()
