from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if metric not in df.columns:
        print(f"[plot] metric not found: {metric}")
        return
    for model, sub in df.groupby("model"):
        ss = sub.copy()
        ss[metric] = pd.to_numeric(ss[metric], errors="coerce")
        ss = ss.dropna(subset=[metric])
        if ss.empty:
            continue
        ss = ss.sort_values("layer")
        plt.figure(figsize=(8, 4))
        plt.plot(ss["layer"].astype(str), ss[metric], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{model}: {metric}")
        plt.xlabel("layer")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}__{metric}.png", dpi=160)
        plt.close()
