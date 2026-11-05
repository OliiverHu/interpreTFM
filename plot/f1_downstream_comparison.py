from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _best_f1_values(df: pd.DataFrame) -> pd.Series:
    if "max_f1" in df.columns:
        return pd.to_numeric(df["max_f1"], errors="coerce").dropna()
    if "best_f1" in df.columns:
        return pd.to_numeric(df["best_f1"], errors="coerce").dropna()
    return pd.to_numeric(df["f1"], errors="coerce").dropna()


def plot_overlay_hist(
    sae_best: pd.DataFrame,
    act_best: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.hist(_best_f1_values(sae_best), bins=40, alpha=0.6, label="SAE")
    plt.hist(_best_f1_values(act_best), bins=40, alpha=0.6, label="Activation dims")
    plt.xlabel("Best F1 per feature")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topk(
    sae_best: pd.DataFrame,
    act_best: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
    topk: int = 100,
) -> None:
    import matplotlib.pyplot as plt

    s = _best_f1_values(sae_best).sort_values(ascending=False).head(topk).to_numpy()
    a = _best_f1_values(act_best).sort_values(ascending=False).head(topk).to_numpy()
    plt.figure(figsize=(7, 4.5))
    if len(s):
        plt.plot(np.arange(1, len(s) + 1), s, marker="o", markersize=3, label="SAE")
    if len(a):
        plt.plot(np.arange(1, len(a) + 1), a, marker="o", markersize=3, label="Activation dims")
    plt.xlabel("Rank")
    plt.ylabel("Best F1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_concept_usage_bar(
    best: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
    topn: int,
) -> None:
    import matplotlib.pyplot as plt

    if best.empty:
        return
    concept_col = "top_term_name" if "top_term_name" in best.columns else ("term_name" if "term_name" in best.columns else "term_id")
    counts = best[concept_col].fillna(best.get("top_term_id", best.get("term_id", ""))).astype(str).value_counts().head(topn)
    plt.figure(figsize=(9, max(4, 0.28 * len(counts) + 1.5)))
    plt.barh(counts.index[::-1], counts.values[::-1])
    plt.xlabel("# features")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grid(table: pd.DataFrame, *, out_path: str | Path, title: str, colorbar_label: str) -> None:
    import matplotlib.pyplot as plt

    arr = table.values
    plt.figure(figsize=(7, 5))
    im = plt.imshow(arr, aspect="auto")
    plt.xticks(range(table.shape[1]), table.columns.tolist())
    plt.yticks(range(table.shape[0]), table.index.tolist())
    plt.xlabel("min_true_pos")
    plt.ylabel("threshold")
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")
    plt.colorbar(im, label=colorbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_combined_concepts_bar(
    combined: pd.DataFrame,
    *,
    cutoff: float,
    out_root: Path,
    safe_tag,
) -> None:
    import matplotlib.pyplot as plt

    col = f"n_concepts_best_f1_ge_{cutoff}"
    if col not in combined.columns:
        return
    pivot = combined.pivot(index="model", columns="source", values=col)
    pivot.to_csv(out_root / f"combined_concepts_best_f1_ge_{safe_tag(cutoff)}.csv")

    x = np.arange(len(pivot.index))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    if "SAE" in pivot.columns:
        plt.bar(x - width / 2, pivot["SAE"].fillna(0).values, width, label="SAE")
    if "activation" in pivot.columns:
        plt.bar(x + width / 2, pivot["activation"].fillna(0).values, width, label="Activation")
    plt.xticks(x, pivot.index.tolist())
    plt.ylabel("# concepts")
    plt.title(f"# concepts with best F1 >= {cutoff}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / f"combined_concepts_best_f1_ge_{safe_tag(cutoff)}.png", dpi=200)
    plt.close()
