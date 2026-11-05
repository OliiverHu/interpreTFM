"""Build a cell x concept-feature AnnData from classification-token activations.

Cell-level activations of one layer are encoded with an SAE, and the SAE features are kept
by their best concept F1 (``per_feature_best.csv``): one row per feature, optionally one
feature per concept, an F1 cutoff and an optional cap. The output AnnData has one column
per kept feature, named by its concept, and keeps ``obs`` and ``obsm["spatial"]`` of the
input.

  python -m interp_pipeline.downstream.f1.build_interpretable_from_sae_latents --help
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.tis.io import encode_sae_latents, extract_cls_from_shards_aligned


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a cell x concept-feature AnnData from classification-token activations and SAE features."
    )
    ap.add_argument("--runs-root", required=True,
                    help="Folder with activations/<layer>/, sae/<layer>/sae_<layer>.pt and f1_analysis/per_feature_best.csv.")
    ap.add_argument("--adata", required=True, help="AnnData (.h5ad) whose cells are matched to the store's example ids.")
    ap.add_argument("--layer", default="layer_4.norm2")
    ap.add_argument("--sae-threshold", type=float, default=0.6, help="Threshold rows of per_feature_best.csv to use.")
    ap.add_argument("--acts-root", default=None, help="Default: <runs-root>/activations/<layer>.")
    ap.add_argument("--sae-ckpt", default=None, help="Default: <runs-root>/sae/<layer>/sae_<layer>.pt.")
    ap.add_argument("--per-feature-best", default=None, help="Default: <runs-root>/f1_analysis/per_feature_best.csv.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <runs-root>/interpretable_adata/<layer>/sae_thr_<sae-threshold>.")
    ap.add_argument("--max-shards", type=int, default=512)
    ap.add_argument("--f1-cutoff", type=float, default=0.60, help="Keep features with best_f1 >= this value.")
    ap.add_argument("--topk-max", type=int, default=None, help="Optional cap on the number of kept features.")
    ap.add_argument("--var-name-mode", choices=["term_name", "term_id"], default="term_name")
    ap.add_argument("--no-dedup-by-concept", dest="dedup_by_concept", action="store_false",
                    help="Keep several features per concept (default: only the best feature per GO id).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8192)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    LAYER = args.layer
    SAE_THRESHOLD = args.sae_threshold
    BASE_ADATA = args.adata
    ACTS_ROOT = args.acts_root or os.path.join(args.runs_root, "activations", LAYER)
    SAE_CKPT = args.sae_ckpt or os.path.join(args.runs_root, "sae", LAYER, f"sae_{LAYER}.pt")
    PER_FEATURE_BEST = args.per_feature_best or os.path.join(args.runs_root, "f1_analysis", "per_feature_best.csv")
    OUTDIR = args.out_dir or os.path.join(args.runs_root, "interpretable_adata", LAYER, f"sae_thr_{SAE_THRESHOLD}")
    MAX_SHARDS = args.max_shards
    F1_CUTOFF = args.f1_cutoff
    TOPK_MAX = args.topk_max
    VAR_NAME_MODE = args.var_name_mode
    DEDUP_BY_CONCEPT = args.dedup_by_concept
    os.makedirs(OUTDIR, exist_ok=True)

    print("[1] Load base AnnData")
    adata = sc.read_h5ad(BASE_ADATA)
    print("  base:", adata.shape)

    print("[2] Extract aligned <CLS> activations")
    A_aligned, has_row = extract_cls_from_shards_aligned(
        ACTS_ROOT,
        obs_names=adata.obs_names.tolist(),
        max_shards=MAX_SHARDS,
    )
    n_ok = int(has_row.sum())
    print(f"  aligned CLS: {n_ok}/{adata.n_obs}")

    # keep only rows that exist
    adata_sub = adata[has_row].copy()
    A_use = A_aligned[has_row]
    print("  A_use:", A_use.shape, "adata_sub:", adata_sub.shape)

    print("[3] Encode SAE latents (cells × K)")
    if not os.path.exists(SAE_CKPT):
        raise FileNotFoundError(f"Missing SAE checkpoint: {SAE_CKPT}")
    Z = encode_sae_latents(A_use, SAE_CKPT, device=args.device, batch_size=args.batch_size)
    print("  Z:", Z.shape)

    print("[4] Load per_feature_best and select latents")
    best = pd.read_csv(PER_FEATURE_BEST)

    needed = {"layer", "threshold", "best_term_id", "best_f1"}
    missing = needed - set(best.columns)
    if missing:
        raise RuntimeError(f"{PER_FEATURE_BEST} missing columns {missing}. have={list(best.columns)}")

    # the latent column is named 'latent' or 'feature'
    latent_col = "latent" if "latent" in best.columns else ("feature" if "feature" in best.columns else None)
    if latent_col is None:
        raise RuntimeError(f"Cannot find latent column in {PER_FEATURE_BEST}. Have: {list(best.columns)}")

    best = best[(best["layer"] == LAYER) & (best["threshold"] == float(SAE_THRESHOLD))].copy()
    if best.empty:
        raise RuntimeError(f"No rows for layer={LAYER}, thr={SAE_THRESHOLD} in {PER_FEATURE_BEST}")

    # Ensure term name exists
    if "best_term_name" not in best.columns:
        best["best_term_name"] = best["best_term_id"].astype(str)

    # (A) best-only per latent (keep highest best_f1 row per latent)
    best = best.sort_values("best_f1", ascending=False)
    best = best.drop_duplicates(subset=[latent_col], keep="first").copy()
    if latent_col != "latent":
        best = best.rename(columns={latent_col: "latent"})

    # (B) dedupe by concept (GO id), keep best latent for each concept
    if DEDUP_BY_CONCEPT:
        best = best.sort_values("best_f1", ascending=False)
        best = best.drop_duplicates(subset=["best_term_id"], keep="first").copy()

    # (C) apply F1 cutoff + optional cap
    best = best[best["best_f1"] >= float(F1_CUTOFF)].copy()
    if TOPK_MAX is not None:
        best = best.sort_values("best_f1", ascending=False).head(int(TOPK_MAX)).copy()

    if best.empty:
        raise RuntimeError(f"No latents pass best_f1 >= {F1_CUTOFF}. Lower cutoff or change threshold.")

    print(
        "  selection summary:",
        f"rows={len(best)}",
        f"unique_latents={best['latent'].nunique()}",
        f"unique_terms={best['best_term_id'].nunique()}",
        f"(dedup_by_concept={DEDUP_BY_CONCEPT})",
    )

    # selected latent indices
    latents = best["latent"].astype(int).to_numpy()
    if latents.max() >= Z.shape[1]:
        raise RuntimeError(f"Selected latent id exceeds Z dims. max_latent={latents.max()} Z.shape={Z.shape}")

    Z_sel = Z[:, latents].astype(np.float32)
    print("  selected Z:", Z_sel.shape, f"(cutoff={F1_CUTOFF}, topk={TOPK_MAX})")

    # var_names: term id or name
    if VAR_NAME_MODE == "term_id":
        var_names = best["best_term_id"].astype(str).to_list()
    else:
        var_names = best["best_term_name"].fillna(best["best_term_id"]).astype(str).to_list()

    # AnnData requires unique var_names
    seen = {}
    uniq = []
    for v in var_names:
        if v not in seen:
            seen[v] = 0
            uniq.append(v)
        else:
            seen[v] += 1
            uniq.append(f"{v}__{seen[v]}")
    var_names = uniq

    print("[5] Create interpretable AnnData")
    out_var = best.rename(columns={"latent": "sae_latent"})[
        ["sae_latent", "best_f1", "best_term_id", "best_term_name", "layer", "threshold"]
    ].reset_index(drop=True)

    out = sc.AnnData(
        X=Z_sel,
        obs=adata_sub.obs.copy(),
        var=out_var,
    )
    out.var_names = var_names

    # keep spatial coords if present
    if "spatial" in adata_sub.obsm:
        out.obsm["spatial"] = adata_sub.obsm["spatial"].copy()

    # write manifest + h5ad
    manifest_path = os.path.join(
        OUTDIR, f"selected_latents_bestonly_conceptdedup_f1cut{F1_CUTOFF}_top{TOPK_MAX if TOPK_MAX else 'all'}.csv"
    )
    out.var.to_csv(manifest_path, index=False)

    out_path = os.path.join(
        OUTDIR,
        f"adata_interpretable_{LAYER}_saeThr{SAE_THRESHOLD}_bestonly_conceptdedup_f1cut{F1_CUTOFF}_top{TOPK_MAX if TOPK_MAX else 'all'}.h5ad",
    )
    out.write_h5ad(out_path)

    print("[OK] wrote:")
    print("  ", out_path)
    print("  ", manifest_path)


if __name__ == "__main__":
    main()