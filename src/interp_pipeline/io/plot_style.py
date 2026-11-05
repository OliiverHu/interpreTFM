from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib as mpl


def apply_publication_style(base_fontsize: float = 10.0) -> None:
    mpl.rcParams.update(
        {
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize + 1.0,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": max(8.0, base_fontsize - 1.0),
            "ytick.labelsize": max(8.0, base_fontsize - 1.0),
            "legend.fontsize": max(8.0, base_fontsize - 1.0),
            "figure.titlesize": base_fontsize + 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def save_figure_bundle(fig, out_path: str | Path, dpi: int = 300) -> Tuple[Path, Path, Path]:
    p = Path(out_path)
    if p.suffix:
        stem = p.with_suffix("")
    else:
        stem = p
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    fig.savefig(svg)
    return png, pdf, svg
