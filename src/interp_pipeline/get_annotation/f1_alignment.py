from __future__ import annotations

"""
interp_pipeline.get_annotation.f1_alignment

Library module for F1 alignment + heldout reporting.

Key guarantees:
- Activation shard index.pt token_ids may be:
    1) Ensembl IDs ("ENSG...")
    2) scGPT vocab indices (often strings of ints, e.g. "60695")
    3) gene symbols
  We robustly map all three to Ensembl IDs before GT lookup.

- Uses sparse accumulation for TP:
    TP = A^T @ G
  where A is (N,K) latent-active boolean, G is (N,T) gene×term CSR.

- Heldout reporting:
    * Default split unit is gene-token rows, not cells and not the scGPT cell token.
    * Only rows with token_unit == "gene" and token IDs that map to GT genes are evaluated.
    * VALID: keep only top-M features per concept per threshold before writing.
    * TEST: evaluate only those selected features (from VALID) per concept per threshold.

Important interpretation:
- This function evaluates SAE feature/concept alignment over heldout gene-token rows.
- It does not use cell-token/CLS-token pooling and does not itself guarantee SAE-training isolation.
"""

import os
import json
import glob
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm


BinarizeMethod = Literal["threshold"]


@dataclass(frozen=True)
class TokenMapper:
    """
    Maps token_ids stored in activation shards -> Ensembl gene IDs.

    decode_token:
      int(vocab_id) -> symbol or ENSG or special token
    sym2ens:
      symbol -> ENSG
    """
    decode_token: Optional[Callable[[int], str]]
    sym2ens: Dict[str, str]
    ens2sym: Optional[Dict[str, str]] = None

    _special: Set[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self._special is None:
            object.__setattr__(self, "_special", {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"})

    @staticmethod
    def _is_int_str(x: Any) -> bool:
        try:
            return str(x).strip().isdigit()
        except Exception:
            return False

    def token_to_ens(self, tid: Any, decode_cache: Dict[int, str]) -> Optional[str]:
        s = str(tid).strip()
        if not s or s in self._special:
            return None

        if s.startswith("ENSG"):
            return s

        if self._is_int_str(s):
            if self.decode_token is None:
                return None
            ii = int(s)
            sym = decode_cache.get(ii)
            if sym is None:
                sym = self.decode_token(ii)
                decode_cache[ii] = sym

            if not sym or sym in self._special:
                return None
            if sym.startswith("ENSG"):
                return sym
            return self.sym2ens.get(sym)

        return self.sym2ens.get(s)

    def token_to_symbol(self, tid: Any, decode_cache: Dict[int, str]) -> Optional[str]:
        s = str(tid).strip()
        if not s or s in self._special:
            return None
        if self.ens2sym is None:
            return None
        if s in self.ens2sym:
            return self.ens2sym.get(s)
        if self._is_int_str(s):
            if self.decode_token is None:
                return None
            ii = int(s)
            sym = decode_cache.get(ii)
            if sym is None:
                sym = self.decode_token(ii)
                decode_cache[ii] = sym
            if not sym or sym in self._special:
                return None
            if sym.startswith("ENSG"):
                return self.ens2sym.get(sym)
            return sym
        return None


def dense_bool_to_csr(A_bool: np.ndarray) -> sparse.csr_matrix:
    if A_bool.dtype != np.bool_:
        A_bool = A_bool.astype(bool, copy=False)
    rows, cols = np.where(A_bool)
    # int32, not int8: scipy keeps int8 for int8 @ int8, so per-shard true-positive counts above 127 wrapped
    data = np.ones(rows.shape[0], dtype=np.int32)
    return sparse.csr_matrix((data, (rows, cols)), shape=A_bool.shape)


def accumulate_tp_sparse_into_dense(tp_dense: np.ndarray, A_bool: np.ndarray, G_csr: sparse.csr_matrix) -> None:
    A_csr = dense_bool_to_csr(A_bool)
    TP = (A_csr.T @ G_csr).tocoo()
    tp_dense[TP.row, TP.col] += TP.data.astype(tp_dense.dtype, copy=False)


def compute_f1_from_counts(
    tp: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp_f = tp.astype(np.float32, copy=False)
    fp = (pred_pos[:, None] - tp).astype(np.int64, copy=False)
    fn = (true_pos[None, :] - tp).astype(np.int64, copy=False)
    fp_f = fp.astype(np.float32, copy=False)
    fn_f = fn.astype(np.float32, copy=False)

    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return precision, recall, f1, fp, fn


def top_hits_per_latent(
    *,
    f1: np.ndarray,
    tp: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    term_ids: Sequence[str],
    topn: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    K, _T = f1.shape
    topn = int(topn)

    for k in range(K):
        best = np.argsort(-f1[k])[:topn]
        for j in best:
            rows.append(
                {
                    "latent": int(k),
                    "threshold": float(threshold),
                    "term_id": str(term_ids[j]),
                    "f1": float(f1[k, j]),
                    "tp": int(tp[k, j]),
                    "pred_pos": int(pred_pos[k]),
                    "true_pos": int(true_pos[j]),
                }
            )
    return rows


def _split_genes(genes: Sequence[str], valid_frac: float, test_frac: float, seed: int) -> Tuple[Set[str], Set[str], Set[str]]:
    rng = np.random.default_rng(int(seed))
    uniq = np.array(sorted(set(genes)), dtype=object)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = int(round(n * float(test_frac)))
    n_valid = int(round(n * float(valid_frac)))

    test = set(uniq[:n_test].tolist())
    valid = set(uniq[n_test:n_test + n_valid].tolist())
    train = set(uniq[n_test + n_valid:].tolist())
    return train, valid, test




def _stable_row_uniform(*, seed: int, shard_name: str, row_index: int, token_id: Any) -> float:
    """
    Deterministic uniform number in [0, 1) for one activation row.

    This is used for gene-token-row heldout. It avoids loading all rows into memory
    and makes the split reproducible independent of shard processing order.
    """
    key = f"{int(seed)}|{str(shard_name)}|{int(row_index)}|{str(token_id)}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    v = int.from_bytes(h, byteorder="little", signed=False)
    return v / float(2 ** 64)


def _assign_gene_token_row_split(
    *,
    seed: int,
    shard_name: str,
    row_index: int,
    token_id: Any,
    valid_frac: float,
    test_frac: float,
) -> int:
    """
    Return split id for a single mapped gene-token row:
      0 = valid
      1 = test
     -1 = train / ignored for valid-test F1 counters
    """
    u = _stable_row_uniform(seed=seed, shard_name=shard_name, row_index=row_index, token_id=token_id)
    test_frac = float(test_frac)
    valid_frac = float(valid_frac)
    if u < test_frac:
        return 1
    if u < test_frac + valid_frac:
        return 0
    return -1


def first_copy_row_mask(example_ids: Sequence[Any], seen: Set[str]) -> np.ndarray:
    """
    Boolean mask of the rows that belong to the first copy of each cell.

    A copy is a contiguous run of rows with the same example (cell) id. Runs of a cell
    already in ``seen`` are dropped;
    ``seen`` is updated in place so that the rule holds across shards. A store without
    duplicated cells gives an all-True mask; stores that hold a cell more than once keep
    the first copy.
    """
    keep = np.ones(len(example_ids), dtype=bool)
    prev: Any = object()
    drop = False
    for i, e in enumerate(example_ids):
        if i == 0 or e != prev:
            key = str(e)
            drop = key in seen
            seen.add(key)
            prev = e
        if drop:
            keep[i] = False
    return keep


def _load_sym2ens_from_cosmx_adata(adata_path: str, symbol_col: str = "index") -> Dict[str, str]:
    import scanpy as sc
    adata = sc.read_h5ad(adata_path)
    if symbol_col not in adata.var.columns:
        raise ValueError(f"Expected adata.var['{symbol_col}'] to contain gene symbols.")
    ens = adata.var.index.astype(str).tolist()
    sym = adata.var[symbol_col].astype(str).tolist()
    return {s: e for e, s in zip(ens, sym)}


def _load_ens2sym_from_adata(
    adata_path: str,
    *,
    ensembl_col_candidates: Sequence[str] = ("ensembl_id", "gene_ids", "gene_id"),
    symbol_col_candidates: Sequence[str] = ("gene_symbol", "gene_name", "index"),
) -> Dict[str, str]:
    import scanpy as sc
    adata = sc.read_h5ad(adata_path)
    ens_col = next((c for c in ensembl_col_candidates if c in adata.var.columns), None)
    sym_col = next((c for c in symbol_col_candidates if c in adata.var.columns), None)
    if ens_col is None or sym_col is None:
        return {}
    ens = adata.var[ens_col].astype(str).tolist()
    sym = adata.var[sym_col].astype(str).tolist()
    return {e: s for e, s in zip(ens, sym)}


def _load_gene_alias_to_gt_gene_from_adata(
    adata_path: str,
    gt_gene_set: Set[str],
    *,
    ensembl_col_candidates: Sequence[str] = ("ensembl_id", "gene_ids", "gene_id"),
    symbol_col_candidates: Sequence[str] = ("gene_symbol", "gene_name", "feature_name", "index"),
) -> Dict[str, str]:
    """
    Build a local alias -> GT gene map from AnnData so token IDs can be matched
    robustly regardless of whether the shard stores ENSG IDs, symbols, or a mix.

    The returned values are always canonical GT row keys, i.e. exact strings from
    gt_gene_set.
    """
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)
    ens_col = next((c for c in ensembl_col_candidates if c in adata.var.columns), None)
    sym_col = next((c for c in symbol_col_candidates if c in adata.var.columns), None)

    alias_to_gt: Dict[str, str] = {}
    if ens_col is None and sym_col is None:
        return alias_to_gt

    ens_vals = adata.var[ens_col].astype(str).tolist() if ens_col is not None else [None] * adata.n_vars
    sym_vals = adata.var[sym_col].astype(str).tolist() if sym_col is not None else [None] * adata.n_vars

    for ens, sym in zip(ens_vals, sym_vals):
        candidates = [c for c in [ens, sym] if c is not None]
        for alias in candidates:
            alias = str(alias).strip()
            if not alias:
                continue
            if alias in gt_gene_set:
                alias_to_gt[alias] = alias
                continue
            # Prefer symbol if it is already a GT row key; otherwise keep ENSG if that is the GT key.
            if sym is not None and str(sym).strip() in gt_gene_set:
                alias_to_gt[alias] = str(sym).strip()
            elif ens is not None and str(ens).strip() in gt_gene_set:
                alias_to_gt[alias] = str(ens).strip()

    return alias_to_gt


def _load_scgpt_decoder(scgpt_ckpt: str) -> Callable[[int], str]:
    vocab_path = os.path.join(scgpt_ckpt, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        if isinstance(vocab, dict):
            id_to_gene = {int(v): str(k) for k, v in vocab.items()}

            def _from_vocab(i: int) -> str:
                return id_to_gene.get(int(i), "")

            return _from_vocab

    from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
    from interp_pipeline.adapters.model_base import ModelSpec

    adapter = ScGPTAdapter()
    handle = adapter.load(ModelSpec(name="scgpt", checkpoint=scgpt_ckpt, device="cpu", options={}))
    tok = handle.tokenizer

    for parent_attr in ["gene_vocab", "vocab"]:
        if hasattr(tok, parent_attr):
            v = getattr(tok, parent_attr)
            for itos_attr in ["itos", "get_itos"]:
                if hasattr(v, itos_attr):
                    itos = getattr(v, itos_attr)
                    itos = itos() if callable(itos) else itos
                    if isinstance(itos, (list, tuple)) and len(itos) > 0:
                        def _fast(i: int, itos_ref=itos) -> str:
                            i = int(i)
                            if i < 0 or i >= len(itos_ref):
                                return ""
                            return str(itos_ref[i])
                        return _fast

    if hasattr(tok, "decode") and callable(getattr(tok, "decode")):
        def _decode_one(i: int) -> str:
            out = tok.decode([[int(i)]])
            if not out:
                return ""
            if isinstance(out[0], list):
                return str(out[0][0]) if out[0] else ""
            return str(out[0]) if out else ""
        return _decode_one

    raise RuntimeError("Cannot build scGPT decoder (no vocab itos / decode).")


class EncoderOnlySAE(torch.nn.Module):
    """
    Minimal SAE encoder used for F1 alignment.

    Only the encoder weights are read, so checkpoints of SAE classes with different
    decoders load the same way. F1 alignment needs the latents z, not the reconstruction.

    Supported checkpoint keys:
      - encoder.weight / W_enc / enc.weight
      - encoder.bias / b_enc / enc.bias, optional
      - bias / x_bias, optional input-centering bias
    """
    def __init__(
        self,
        w_enc: torch.Tensor,
        b_enc: Optional[torch.Tensor] = None,
        x_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("w_enc", w_enc.float().contiguous())
        self.register_buffer("b_enc", None if b_enc is None else b_enc.float().contiguous())
        self.register_buffer("x_bias", None if x_bias is None else x_bias.float().contiguous())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        if self.x_bias is not None:
            x = x - self.x_bias.to(device=x.device, dtype=x.dtype)
        z = x @ self.w_enc.to(device=x.device, dtype=x.dtype).T
        if self.b_enc is not None:
            z = z + self.b_enc.to(device=x.device, dtype=x.dtype)
        z = torch.relu(z)
        return x, z


def _get_first_tensor(state: Dict[str, torch.Tensor], keys: Sequence[str]) -> Optional[torch.Tensor]:
    for k in keys:
        v = state.get(k)
        if torch.is_tensor(v):
            return v
    return None


def _load_sae(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"SAE checkpoint must be dict-like. Got {type(ckpt)}")

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt.get("model", None)))
    if state is None and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"Could not find state_dict/model_state_dict/model in {ckpt_path}")

    w_enc = _get_first_tensor(state, ["encoder.weight", "W_enc", "enc.weight"])
    if w_enc is None:
        raise RuntimeError(
            "Could not find encoder weight in SAE checkpoint. "
            f"First keys: {list(state.keys())[:30]}"
        )

    w_enc = w_enc.float()
    if w_enc.ndim != 2:
        raise RuntimeError(f"Expected encoder weight to be 2D, got shape={tuple(w_enc.shape)}")

    n_lat, d_in_from_w = int(w_enc.shape[0]), int(w_enc.shape[1])
    d_in = int(ckpt.get("d_in", d_in_from_w))
    n_lat_ckpt = int(ckpt.get("n_latents", n_lat))

    if d_in != d_in_from_w:
        raise RuntimeError(f"Checkpoint d_in={d_in}, but encoder weight has input dim={d_in_from_w}")
    if n_lat_ckpt != n_lat:
        raise RuntimeError(f"Checkpoint n_latents={n_lat_ckpt}, but encoder weight has n_latents={n_lat}")

    b_enc = _get_first_tensor(state, ["encoder.bias", "b_enc", "enc.bias"])
    x_bias = _get_first_tensor(state, ["bias", "x_bias"])

    if b_enc is not None and int(b_enc.numel()) != n_lat:
        raise RuntimeError(f"encoder bias has shape={tuple(b_enc.shape)}, expected {n_lat}")
    if x_bias is not None and int(x_bias.numel()) != d_in:
        raise RuntimeError(f"x/input bias has shape={tuple(x_bias.shape)}, expected {d_in}")

    model = EncoderOnlySAE(w_enc=w_enc, b_enc=b_enc, x_bias=x_bias).to(device).eval()

    print(
        "[load_sae] encoder-only loader:",
        f"d_in={d_in}",
        f"n_latents={n_lat}",
        f"has_encoder_bias={b_enc is not None}",
        f"has_x_bias={x_bias is not None}",
    )
    return model, d_in, n_lat


@torch.no_grad()
def _sae_encode(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], torch.Tensor):
        return out[1]
    raise RuntimeError("Unexpected SAE forward output; expected (x_hat, z).")


def heldout_report_for_layer(
    *,
    layer: str,
    store_root: str,
    gt_csv: str,
    sae_ckpt_path: str,
    out_dir: str,
    latent_thresholds: Sequence[float],
    valid_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 0,
    heldout_unit: str = "gene_token_row",
    topM_valid_per_concept_per_threshold: int = 200,
    batch_size: int = 8192,
    device: str = "cuda",
    tp_dtype: Any = np.int32,
    adata_path: Optional[str] = None,
    scgpt_ckpt: Optional[str] = None,
    adata_symbol_col: str = "index",
    dev_mode: bool = False,
    dev_max_shards: Optional[int] = 50,
    dev_max_rows_per_split_per_shard: Optional[int] = 4096,
    dev_only_valid: bool = False,
    dedup_cells: bool = False,
) -> None:
    """
    Heldout evaluation for one layer. Writes:
      - valid_concept_f1_scores.csv   (topM per concept per threshold)
      - test_concept_f1_scores.csv    (restricted to valid-selected features)
      - selected_features_valid.json
      - counts_summary.json
      - heldout_gene_split.json
      - concept_support_by_split.csv

    IMPORTANT: If token_ids in shards are vocab ids, pass adata_path and scgpt_ckpt.

    dedup_cells (default False): keep only the first copy of each cell (``first_copy_row_mask``
    over index.pt ``example_ids``) and ignore the rest. Kept rows keep their shard name and row
    index, so their split assignment is unchanged.
    """
    os.makedirs(out_dir, exist_ok=True)

    heldout_unit = str(heldout_unit)
    if heldout_unit not in {"gene_token_row", "gene_identity"}:
        raise ValueError(
            "heldout_unit must be 'gene_token_row' or 'gene_identity', "
            f"got {heldout_unit!r}"
        )

    gt_df = pd.read_csv(gt_csv, index_col=0)
    gt_genes = gt_df.index.astype(str).tolist()
    gt_gene_set = set(gt_genes)
    term_ids = gt_df.columns.astype(str).tolist()
    GT = gt_df.values.astype(np.int8, copy=False)
    gt_row_for_gene = {g: i for i, g in enumerate(gt_genes)}
    T_terms = GT.shape[1]

    sym2ens: Dict[str, str] = {}
    ens2sym: Dict[str, str] = {}
    alias_to_gt_gene: Dict[str, str] = {}
    decode_token: Optional[Callable[[int], str]] = None

    if adata_path is not None:
        sym2ens = _load_sym2ens_from_cosmx_adata(adata_path, symbol_col=adata_symbol_col)
        ens2sym = _load_ens2sym_from_adata(adata_path)
        alias_to_gt_gene = _load_gene_alias_to_gt_gene_from_adata(adata_path, gt_gene_set)
    if scgpt_ckpt is not None:
        decode_token = _load_scgpt_decoder(scgpt_ckpt)

    mapper = TokenMapper(decode_token=decode_token, sym2ens=sym2ens, ens2sym=ens2sym)
    decode_cache: Dict[int, str] = {}

    acts_root = os.path.join(store_root, "activations", layer)
    shards = sorted([p for p in glob.glob(os.path.join(acts_root, "shard_*")) if os.path.isdir(p)])
    if not shards:
        raise RuntimeError(f"No activation shards found under {acts_root}")

    if dev_mode and dev_max_shards is not None:
        shards = shards[: int(dev_max_shards)]

    # Discover all observed genes by scanning every row of every shard (not a prefix), so that the
    # split is independent of row order.
    observed_genes: List[str] = []
    observed_gene_counts: Dict[str, int] = {}
    keep_by_shard: Dict[str, np.ndarray] = {}
    seen_cells: Set[str] = set()
    rows_dropped_duplicate_cells = 0

    for shard in shards:
        idx_path = os.path.join(shard, "index.pt")
        acts_path = os.path.join(shard, "activations.pt")
        if not (os.path.exists(idx_path) and os.path.exists(acts_path)):
            continue
        idx = torch.load(idx_path, map_location="cpu")
        if idx.get("token_unit", "unknown") != "gene":
            continue
        token_ids = idx.get("token_ids", None)
        if token_ids is None:
            continue

        keep = None
        if dedup_cells:
            example_ids = idx.get("example_ids", None)
            if example_ids is None or len(example_ids) != len(token_ids):
                raise RuntimeError(f"dedup_cells needs example_ids aligned with token_ids in {idx_path}")
            keep = first_copy_row_mask(example_ids, seen_cells)
            keep_by_shard[shard] = keep
            rows_dropped_duplicate_cells += int((~keep).sum())

        for i, tid in enumerate(token_ids):
            if keep is not None and not keep[i]:
                continue
            candidates: List[str] = []
            tid_str = str(tid).strip()
            if tid_str:
                candidates.append(tid_str)

            ens = mapper.token_to_ens(tid, decode_cache)
            if ens is not None:
                candidates.append(str(ens).strip())

            sym = mapper.token_to_symbol(tid, decode_cache)
            if sym is not None:
                candidates.append(str(sym).strip())

            seen_candidates: Set[str] = set()
            matched_gene: Optional[str] = None
            for cand in candidates:
                if not cand or cand in seen_candidates:
                    continue
                seen_candidates.add(cand)

                if cand in gt_gene_set:
                    matched_gene = cand
                    break

                alias_hit = alias_to_gt_gene.get(cand)
                if alias_hit is not None and alias_hit in gt_gene_set:
                    matched_gene = alias_hit
                    break

            if matched_gene is not None:
                observed_genes.append(matched_gene)
                observed_gene_counts[matched_gene] = observed_gene_counts.get(matched_gene, 0) + 1

    if not observed_genes:
        debug_path = os.path.join(out_dir, "heldout_gene_mapping_debug.json")
        debug_payload = {
            "gt_n_genes": int(len(gt_genes)),
            "gt_examples": gt_genes[:20],
            "has_adata_path": adata_path is not None,
            "has_scgpt_ckpt": scgpt_ckpt is not None,
            "n_sym2ens": int(len(sym2ens)),
            "n_ens2sym": int(len(ens2sym)),
            "n_alias_to_gt_gene": int(len(alias_to_gt_gene)),
        }
        try:
            with open(debug_path, "w") as f:
                json.dump(debug_payload, f, indent=2)
        except Exception:
            pass
        raise RuntimeError(
            "No genes observed in shards that map to GT rows. "
            "If the token_ids are vocabulary ids, pass adata_path+scgpt_ckpt."
        )

    train_genes: Set[str] = set()
    valid_genes: Set[str] = set()
    test_genes: Set[str] = set()
    if heldout_unit == "gene_identity":
        train_genes, valid_genes, test_genes = _split_genes(
            observed_genes,
            valid_frac=valid_frac,
            test_frac=test_frac,
            seed=seed,
        )

    split_meta: Dict[str, Any] = {
        "split_type": heldout_unit,
        "valid_frac": float(valid_frac),
        "test_frac": float(test_frac),
        "seed": int(seed),
        "n_observed_genes": int(len(set(observed_genes))),
        "n_observed_gene_token_rows_mapped_to_gt": int(len(observed_genes)),
        "observed_gene_count_min": int(min(observed_gene_counts.values())) if observed_gene_counts else 0,
        "observed_gene_count_median": float(np.median(list(observed_gene_counts.values()))) if observed_gene_counts else 0.0,
        "observed_gene_count_max": int(max(observed_gene_counts.values())) if observed_gene_counts else 0,
    }
    if dedup_cells:
        split_meta.update(
            {
                "dedup_cells": True,
                "rows_dropped_duplicate_cells": int(rows_dropped_duplicate_cells),
                "n_cells_kept": int(len(seen_cells)),
            }
        )
    if heldout_unit == "gene_identity":
        split_meta.update(
            {
                "note": (
                    "Genes are split into train/valid/test by Ensembl ID. "
                    "Gene-token rows inherit split membership from mapped gene identity."
                ),
                "n_train_genes": int(len(train_genes)),
                "n_valid_genes": int(len(valid_genes)),
                "n_test_genes": int(len(test_genes)),
                "n_observed_gene_token_rows_mapped_to_gt": int(len(observed_genes)),
                "train_genes": sorted(train_genes),
                "valid_genes": sorted(valid_genes),
                "test_genes": sorted(test_genes),
            }
        )
    else:
        split_meta.update(
            {
                "note": (
                    "Gene-token activation rows are split directly by deterministic row-level hash. "
                    "Rows are kept only when token_unit == 'gene' and token_id maps to a GT gene. "
                    "Special/cell tokens such as scGPT 60694/60695 do not participate."
                ),
            }
        )

    with open(os.path.join(out_dir, "heldout_split.json"), "w") as f:
        json.dump(split_meta, f, indent=2)
    # The same split metadata is also written as heldout_gene_split.json.
    with open(os.path.join(out_dir, "heldout_gene_split.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    sae, d_in, K = _load_sae(sae_ckpt_path, device=dev)

    thresholds = [float(x) for x in latent_thresholds]
    splits = ("valid",) if (dev_mode and dev_only_valid) else ("valid", "test")

    tp: Dict[str, Dict[float, np.ndarray]] = {
        sp: {thr: np.zeros((K, T_terms), dtype=tp_dtype) for thr in thresholds}
        for sp in splits
    }
    pred_pos: Dict[str, Dict[float, np.ndarray]] = {
        sp: {thr: np.zeros((K,), dtype=np.int64) for thr in thresholds}
        for sp in splits
    }
    true_pos: Dict[str, np.ndarray] = {sp: np.zeros((T_terms,), dtype=np.int64) for sp in splits}

    rows_total = 0
    rows_mapped_to_gt = 0
    rows_by_split: Dict[str, int] = {sp: 0 for sp in splits}
    mapped_rows_by_split: Dict[str, int] = {sp: 0 for sp in splits}

    split_iter = (("valid", 0),) if (dev_mode and dev_only_valid) else (("valid", 0), ("test", 1))

    for shard in tqdm(shards, desc=f"heldout:{layer}"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            continue
        idx = torch.load(idx_path, map_location="cpu")
        if idx.get("token_unit", "unknown") != "gene":
            continue
        token_ids = idx.get("token_ids", None)
        if token_ids is None or len(token_ids) != X.shape[0]:
            continue

        rows_total += int(X.shape[0])

        gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
        row_split = np.full((X.shape[0],), -1, dtype=np.int8)
        keep = keep_by_shard.get(shard) if dedup_cells else None
        if dedup_cells and keep is None:
            raise RuntimeError(f"dedup_cells: no keep mask for {shard}")

        for i, tid in enumerate(token_ids):
            if keep is not None and not keep[i]:
                continue  # later copy of a cell: gt_rows stays -1, so the row is ignored
            candidates: List[str] = []
            tid_str = str(tid).strip()
            if tid_str:
                candidates.append(tid_str)

            ens = mapper.token_to_ens(tid, decode_cache)
            if ens is not None:
                candidates.append(str(ens).strip())

            sym = mapper.token_to_symbol(tid, decode_cache)
            if sym is not None:
                candidates.append(str(sym).strip())

            seen_candidates: Set[str] = set()
            r: Optional[int] = None
            matched_gene: Optional[str] = None
            for cand in candidates:
                if not cand or cand in seen_candidates:
                    continue
                seen_candidates.add(cand)
                if cand in gt_row_for_gene:
                    r = gt_row_for_gene.get(cand)
                    matched_gene = cand
                    break
                alias_hit = alias_to_gt_gene.get(cand)
                if alias_hit is not None and alias_hit in gt_row_for_gene:
                    r = gt_row_for_gene.get(alias_hit)
                    matched_gene = alias_hit
                    break

            if r is None:
                continue
            gt_rows[i] = r
            if heldout_unit == "gene_identity":
                # The gene split is keyed by the matched GT gene identifier (a symbol for
                # symbol-keyed GT), not by the Ensembl ID.
                if matched_gene in valid_genes:
                    row_split[i] = 0
                elif matched_gene in test_genes:
                    row_split[i] = 1
            else:
                row_split[i] = _assign_gene_token_row_split(
                    seed=seed,
                    shard_name=os.path.basename(shard),
                    row_index=i,
                    token_id=tid,
                    valid_frac=valid_frac,
                    test_frac=test_frac,
                )

        mask_any = gt_rows >= 0
        if not np.any(mask_any):
            continue
        rows_mapped_to_gt += int(mask_any.sum())

        for sp, sp_id in split_iter:
            m = (row_split == sp_id) & mask_any
            if not np.any(m):
                continue

            if dev_mode and dev_max_rows_per_split_per_shard is not None:
                idxs = np.flatnonzero(m)
                if idxs.size > int(dev_max_rows_per_split_per_shard):
                    idxs = idxs[: int(dev_max_rows_per_split_per_shard)]
                    m2 = np.zeros_like(m, dtype=bool)
                    m2[idxs] = True
                    m = m2

            rows_by_split[sp] += int(m.sum())
            mapped_rows_by_split[sp] += int(m.sum())

            X_use = X[m].float()
            if X_use.shape[1] != d_in:
                raise RuntimeError(f"SAE d_in={d_in} but activations have H={X_use.shape[1]} for layer={layer}")

            G_use = GT[gt_rows[m]]
            G_csr = sparse.csr_matrix(G_use)
            true_pos[sp] += np.asarray(G_csr.sum(axis=0)).ravel().astype(np.int64)

            Z_chunks: List[torch.Tensor] = []
            for i0 in range(0, X_use.shape[0], int(batch_size)):
                xb = X_use[i0:i0 + int(batch_size)].to(dev, non_blocking=True)
                zb = _sae_encode(sae, xb).detach().cpu()
                Z_chunks.append(zb)
            Z = torch.cat(Z_chunks, dim=0).numpy()

            for thr in thresholds:
                A_bool = (Z > float(thr))
                pred_pos[sp][thr] += A_bool.sum(axis=0).astype(np.int64)
                accumulate_tp_sparse_into_dense(tp[sp][thr], A_bool, G_csr)

    def _make_rows_for_split(
        split_name: str,
        restrict: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ) -> pd.DataFrame:
        all_rows: List[Dict[str, Any]] = []

        for thr in thresholds:
            precision, recall, f1, fp, fn = compute_f1_from_counts(
                tp=tp[split_name][thr],
                pred_pos=pred_pos[split_name][thr],
                true_pos=true_pos[split_name],
            )

            if restrict is not None:
                term_to_feats = restrict.get(str(thr), {})
            else:
                term_to_feats = None

            for j, term_id in enumerate(term_ids):
                if term_to_feats is not None:
                    feats = term_to_feats.get(term_id, [])
                    if not feats:
                        continue
                else:
                    feats = list(range(K))

                feats_sorted = sorted(
                    feats,
                    key=lambda k: (float(f1[k, j]), int(tp[split_name][thr][k, j])),
                    reverse=True,
                )
                if split_name == "valid":
                    feats_sorted = feats_sorted[: int(topM_valid_per_concept_per_threshold)]

                for k in feats_sorted:
                    all_rows.append(
                        {
                            "concept": term_id,
                            "feature": int(k),
                            "threshold_pct": float(thr),
                            "precision": float(precision[k, j]),
                            "recall": float(recall[k, j]),
                            "f1": float(f1[k, j]),
                            "tp": int(tp[split_name][thr][k, j]),
                            "fp": int(fp[k, j]),
                            "fn": int(fn[k, j]),
                            "pred_pos": int(pred_pos[split_name][thr][k]),
                            "true_pos": int(true_pos[split_name][j]),
                        }
                    )

        return pd.DataFrame(all_rows)

    df_valid = _make_rows_for_split("valid", restrict=None)

    selected: Dict[str, Dict[str, List[int]]] = {}
    for thr in thresholds:
        thr_key = str(thr)
        selected[thr_key] = {}
        df_thr = df_valid[df_valid["threshold_pct"] == float(thr)]
        if df_thr.empty:
            continue
        for term_id, g in df_thr.groupby("concept"):
            gg = g.sort_values(["f1", "tp"], ascending=False)
            selected[thr_key][term_id] = gg["feature"].astype(int).tolist()

    if dev_mode and dev_only_valid:
        df_test = pd.DataFrame(columns=df_valid.columns)
    else:
        df_test = _make_rows_for_split("test", restrict=selected)

    # Split-level concept support diagnostics.
    support_rows: List[Dict[str, Any]] = []
    for sp in splits:
        for j, term_id in enumerate(term_ids):
            support_rows.append(
                {
                    "split": sp,
                    "concept": str(term_id),
                    "true_pos": int(true_pos[sp][j]),
                }
            )
    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(os.path.join(out_dir, "concept_support_by_split.csv"), index=False)

    df_valid.to_csv(os.path.join(out_dir, "valid_concept_f1_scores.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test_concept_f1_scores.csv"), index=False)
    with open(os.path.join(out_dir, "selected_features_valid.json"), "w") as f:
        json.dump(selected, f, indent=2)

    concept_support_summary = {}
    for sp in splits:
        vals = true_pos[sp]
        concept_support_summary[sp] = {
            "n_concepts": int(vals.size),
            "n_concepts_true_pos_eq_0": int((vals == 0).sum()),
            "n_concepts_true_pos_le_1": int((vals <= 1).sum()),
            "n_concepts_true_pos_le_2": int((vals <= 2).sum()),
            "n_concepts_true_pos_le_3": int((vals <= 3).sum()),
            "n_concepts_true_pos_le_5": int((vals <= 5).sum()),
            "true_pos_min": int(vals.min()) if vals.size else 0,
            "true_pos_median": float(np.median(vals)) if vals.size else 0.0,
            "true_pos_mean": float(np.mean(vals)) if vals.size else 0.0,
            "true_pos_max": int(vals.max()) if vals.size else 0,
        }

    dedup_meta: Dict[str, Any] = {}
    if dedup_cells:
        dedup_meta = {
            "dedup_cells": True,
            "rows_dropped_duplicate_cells": int(rows_dropped_duplicate_cells),
            "n_cells_kept": int(len(seen_cells)),
        }

    with open(os.path.join(out_dir, "counts_summary.json"), "w") as f:
        json.dump(
            {
                "layer": layer,
                "split_type": heldout_unit,
                "n_terms": int(T_terms),
                "n_latents": int(K),
                "thresholds": thresholds,
                "valid_frac": float(valid_frac),
                "test_frac": float(test_frac),
                "seed": int(seed),
                "n_observed_genes": int(len(set(observed_genes))),
                "n_train_genes": int(len(train_genes)),
                "n_valid_genes": int(len(valid_genes)),
                "n_test_genes": int(len(test_genes)),
                "rows_total": int(rows_total),
                "rows_mapped_to_gt": int(rows_mapped_to_gt),
                "rows_by_split": {k: int(v) for k, v in rows_by_split.items()},
                "mapped_rows_by_split": {k: int(v) for k, v in mapped_rows_by_split.items()},
                "concept_support_summary": concept_support_summary,
                "topM_valid_per_concept_per_threshold": int(topM_valid_per_concept_per_threshold),
                "dev_mode": bool(dev_mode),
                "dev_max_shards": None if dev_max_shards is None else int(dev_max_shards),
                "dev_max_rows_per_split_per_shard": None if dev_max_rows_per_split_per_shard is None else int(dev_max_rows_per_split_per_shard),
                "dev_only_valid": bool(dev_only_valid),
                **dedup_meta,
                "note": (
                    "token_ids may be ENSG, vocab-id, or symbol; mapping handled by TokenMapper. "
                    "F1 uses gene-token activation rows. heldout_unit controls whether the split is "
                    "directly over rows or over mapped gene identities."
                ),
            },
            f,
            indent=2,
        )
