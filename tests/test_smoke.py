"""Smoke tests: package imports, CLI entry points, and the README quick-start on toy data.

Everything here runs on CPU without private data or model weights.
"""
import importlib
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import interp_pipeline

ROOT = Path(__file__).resolve().parents[1]

# Optional third-party packages: modules that need them are skipped when they are absent.
OPTIONAL_DEPS = {"transformers", "nnsight", "datasets", "geneformer", "goatools"}
MODULES = sorted(m.name for m in pkgutil.walk_packages(interp_pipeline.__path__, prefix="interp_pipeline."))


def test_version():
    assert interp_pipeline.__version__ == "0.9.0"


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in OPTIONAL_DEPS:
            pytest.skip(f"optional dependency not installed: {e.name}")
        raise


@pytest.mark.parametrize(
    "module",
    [
        "interp_pipeline.layer_experiments.all_layer_sae_f1",
        "interp_pipeline.layer_experiments.aggregate_sae_f1",
        "interp_pipeline.tis.single_layer_screen",
        "interp_pipeline.downstream.f1.build_interpretable_from_sae_latents",
    ],
)
def test_cli_help(module):
    out = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout


def test_quickstart_plan_jobs(tmp_path):
    """README: plan the all-layer jobs from the example model specs without running them."""
    out_root = tmp_path / "runs" / "demo"
    cmd = [
        sys.executable, "-m", "interp_pipeline.layer_experiments.all_layer_sae_f1",
        "--model-spec-json", str(ROOT / "configs" / "model_specs.example.json"),
        "--out-root", str(out_root),
        "--status-only",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 29  # 12 scGPT + 8 Geneformer + 9 C2S-Scale layers
    assert (out_root / f"status_{manifest['run_tag']}.csv").exists()


def test_backend_stages_stop_with_clear_message(tmp_path):
    """Stages whose backend modules are not part of 0.9 stop with a message naming the missing option."""
    cmd = [
        sys.executable, "-m", "interp_pipeline.layer_experiments.all_layer_sae_f1",
        "--model-spec-json", str(ROOT / "configs" / "model_specs.example.json"),
        "--out-root", str(tmp_path / "all_layers"),
        "--stages", "sae", "qc",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode != 0
    assert "--train-script" in out.stderr and "--qc-script" in out.stderr

    cmd = [
        sys.executable, "-m", "interp_pipeline.tis.single_layer_screen",
        "--out-root", str(tmp_path / "tis"), "--models", "scgpt",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode != 0
    assert "--backend-script" in out.stderr


def _toy_store(root: Path, layer: str, n_cells: int, d: int, rng, cls_token: str = "60695"):
    """Write a one-shard activation store with one CLS row and a few gene-token rows per cell."""
    genes = [f"ENSG{i:011d}" for i in range(8)]
    ex, tok, rows = [], [], []
    for c in range(n_cells):
        cell = f"cell{c}"
        ex.append(cell)
        tok.append(cls_token)
        for g in rng.choice(len(genes), size=3, replace=False):
            ex.append(cell)
            tok.append(genes[g])
    X = torch.from_numpy(rng.normal(size=(len(ex), d)).astype(np.float32))
    shard = root / "activations" / layer / "shard_0"
    shard.mkdir(parents=True)
    torch.save(X, shard / "activations.pt")
    torch.save({"example_ids": ex, "token_ids": tok, "token_unit": "gene"}, shard / "index.pt")
    return genes


def test_quickstart_tis(tmp_path):
    """README/usage: TIS of one layer from the Python API, with the shuffled-activation control."""
    import anndata as ad

    from interp_pipeline.tis.core import TISConfig, compute_tis_mis_quantile_streamed, shuffle_activations
    from interp_pipeline.tis.io import build_judge_matrix, extract_cls_from_shards_aligned

    rng = np.random.default_rng(0)
    n_cells, d = 200, 6
    store = tmp_path / "store"
    _toy_store(store, "layer_1.norm2", n_cells, d, rng)
    adata = ad.AnnData(
        X=rng.poisson(2.0, size=(n_cells, 30)).astype(np.float32) + 1.0,
        obs=pd.DataFrame(index=[f"cell{c}" for c in range(n_cells)]),
    )

    A, has_row = extract_cls_from_shards_aligned(
        str(store / "activations" / "layer_1.norm2"), obs_names=adata.obs_names
    )
    assert has_row.all() and A.shape == (n_cells, d)
    J = build_judge_matrix(adata[has_row], mode="log1p_cp10k")
    cfg = TISConfig(K=8, n_trials=50, seed=42)
    tis = compute_tis_mis_quantile_streamed(A[has_row], J, q_low=0.25, q_high=0.75, cfg=cfg)
    tis_null = compute_tis_mis_quantile_streamed(shuffle_activations(A[has_row], seed=0), J, 0.25, 0.75, cfg)
    assert tis.shape == tis_null.shape == (d,)
    assert np.all((tis >= 0) & (tis <= 1))


def test_quickstart_concept_alignment_with_baselines(tmp_path):
    """Best practice: score an SAE, raw hidden dimensions and an untrained SAE identically."""
    from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer
    from interp_pipeline.sae.sae_base import AutoEncoder

    rng = np.random.default_rng(1)
    d, n_latents = 6, 24
    store = tmp_path / "store"
    genes = _toy_store(store, "layer_1.norm2", 80, d, rng)
    gt = pd.DataFrame(rng.integers(0, 2, size=(len(genes), 5)), index=genes, columns=[f"GO:{j:07d}" for j in range(5)])
    gt_csv = tmp_path / "gene_by_concept.csv"
    gt.to_csv(gt_csv)

    torch.manual_seed(0)
    eye = torch.eye(d)
    ckpts = {
        # raw hidden dimensions, split into positive and negative parts
        "raw": {"state_dict": {"encoder.weight": torch.cat([eye, -eye])}, "d_in": d, "n_latents": 2 * d},
        # untrained SAE of the same width as the trained one
        "untrained": {"state_dict": AutoEncoder(d, n_latents).state_dict(), "d_in": d, "n_latents": n_latents},
    }
    for name, ckpt in ckpts.items():
        path = tmp_path / f"{name}.pt"
        torch.save(ckpt, path)
        for unit in ("gene_token_row", "gene_identity"):
            out = tmp_path / "f1" / name / unit
            heldout_report_for_layer(
                layer="layer_1.norm2", store_root=str(store), gt_csv=str(gt_csv), sae_ckpt_path=str(path),
                out_dir=str(out), latent_thresholds=[0.0, 0.15], valid_frac=0.2, test_frac=0.2,
                heldout_unit=unit, device="cpu",
            )
            assert (out / "test_concept_f1_scores.csv").exists()
            assert (out / "counts_summary.json").exists()


def test_usage_niche_sweep():
    """Usage guide: niche sweep on a cell x feature matrix with coordinates."""
    from interp_pipeline.downstream.context.sweep import SweepConfig, run_niche_sweep

    rng = np.random.default_rng(2)
    n = 600
    coords = rng.uniform(0, 1000, size=(n, 2))
    X = (rng.normal(size=(n, 8)) + 2.0 * (coords[:, :1] > 500)).astype(np.float32)
    cfg = SweepConfig(
        radii=[100.0, 150.0], n_clusters_list=[3], method_list=["kmeans", "gmm"], space_list=["m", "xm"],
        seeds=[0, 1, 2], min_cluster_size=20, n_null=5,
    )
    table = run_niche_sweep(X, coords, cfg)
    assert len(table) == 8
    assert {"radius", "space", "method", "n_clusters", "ari_mean", "purity_z", "separability", "score"} <= set(table.columns)
