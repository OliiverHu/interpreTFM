"""F1 de-duplication of cells that appear more than once in an activation store (toy stores, CPU).

  python -m pytest tests/test_f1_dedup_cells.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interp_pipeline.get_annotation.f1_alignment import first_copy_row_mask, heldout_report_for_layer  # noqa: E402


def test_mask_keeps_first_copy_only():
    one = ["a", "a", "b", "b", "b", "c"]
    seen = set()
    keep = first_copy_row_mask(one + one, seen)
    assert keep.tolist() == [True] * 6 + [False] * 6
    assert seen == {"a", "b", "c"}
    # cells already kept in an earlier shard are dropped in later shards
    keep2 = first_copy_row_mask(["c", "d", "d"], seen)
    assert keep2.tolist() == [False, True, True]


def test_mask_leaves_non_duplicated_store_unchanged():
    ex = ["a", "a", "b", "c", "c", "c"]
    assert first_copy_row_mask(ex, set()).all()
    assert first_copy_row_mask([], set()).size == 0


def _write_store(root: Path, shards):
    for name, (ex, tok, X) in shards.items():
        d = root / "activations" / "layer_x" / name
        d.mkdir(parents=True)
        torch.save({"example_ids": ex, "token_ids": tok, "token_unit": "gene"}, d / "index.pt")
        torch.save(X, d / "activations.pt")


def _run(tmp: Path, store: Path, ckpt: Path, gt: Path, tag: str, dedup: bool) -> Path:
    out = tmp / tag
    heldout_report_for_layer(
        layer="layer_x", store_root=str(store), gt_csv=str(gt), sae_ckpt_path=str(ckpt), out_dir=str(out),
        latent_thresholds=[0.0, 0.15], valid_frac=0.3, test_frac=0.3, seed=0, heldout_unit="gene_token_row",
        device="cpu", dev_mode=False, dedup_cells=dedup,
    )
    return out


@pytest.fixture()
def toy(tmp_path):
    rng = np.random.default_rng(0)
    genes = [f"ENSG{i:011d}" for i in range(6)]
    gt = pd.DataFrame(rng.integers(0, 2, size=(6, 4)), index=genes, columns=[f"T{j}" for j in range(4)])
    gt.iloc[:, 0] = 1
    gt_csv = tmp_path / "gt.csv"
    gt.to_csv(gt_csv)
    d = 3
    eye = torch.eye(d)
    ckpt = tmp_path / "raw.pt"
    torch.save({"state_dict": {"encoder.weight": torch.cat([eye, -eye])}, "d_in": d, "n_latents": 2 * d}, ckpt)

    def shard(n_cells, offset):
        ex, tok = [], []
        for c in range(n_cells):
            for g in rng.choice(6, size=int(rng.integers(2, 6)), replace=False):
                ex.append(f"cell{offset + c}")
                tok.append(genes[g])
        X = torch.randn(len(ex), d).half()
        return ex, tok, X

    single = {"shard_0": shard(40, 0), "shard_1": shard(30, 100)}
    # duplicated store: every shard holds each of its cells two times; the second copy differs slightly
    doubled = {k: (ex + ex, tok + tok, torch.cat([X, (X.float() + 1e-3).half()])) for k, (ex, tok, X) in single.items()}
    s1, s2 = tmp_path / "single", tmp_path / "doubled"
    _write_store(s1, single)
    _write_store(s2, doubled)
    return tmp_path, s1, s2, ckpt, gt_csv


def _tables(out: Path):
    return [pd.read_csv(out / f) for f in ("valid_concept_f1_scores.csv", "test_concept_f1_scores.csv",
                                            "concept_support_by_split.csv")]


def test_dedup_on_duplicated_store_equals_single_copy_store(toy):
    tmp, single, doubled, ckpt, gt = toy
    ref = _run(tmp, single, ckpt, gt, "ref", dedup=False)
    ded = _run(tmp, doubled, ckpt, gt, "ded", dedup=True)
    dup = _run(tmp, doubled, ckpt, gt, "dup", dedup=False)
    for a, b in zip(_tables(ref), _tables(ded)):
        pd.testing.assert_frame_equal(a, b)
    c_ref = json.loads((ref / "counts_summary.json").read_text())
    c_ded = json.loads((ded / "counts_summary.json").read_text())
    c_dup = json.loads((dup / "counts_summary.json").read_text())
    assert c_ded["rows_mapped_to_gt"] == c_ref["rows_mapped_to_gt"]
    assert c_ded["rows_by_split"] == c_ref["rows_by_split"]
    assert c_ded["rows_dropped_duplicate_cells"] == c_ref["rows_total"] and c_ded["n_cells_kept"] == 70
    assert c_dup["rows_mapped_to_gt"] == 2 * c_ref["rows_mapped_to_gt"]
    assert "dedup_cells" not in c_ref and "dedup_cells" not in c_dup  # default output unchanged


def test_dedup_on_non_duplicated_store_is_a_no_op(toy):
    tmp, single, _, ckpt, gt = toy
    ref = _run(tmp, single, ckpt, gt, "ref", dedup=False)
    ded = _run(tmp, single, ckpt, gt, "ded1", dedup=True)
    for a, b in zip(_tables(ref), _tables(ded)):
        pd.testing.assert_frame_equal(a, b)
    assert json.loads((ded / "counts_summary.json").read_text())["rows_dropped_duplicate_cells"] == 0
