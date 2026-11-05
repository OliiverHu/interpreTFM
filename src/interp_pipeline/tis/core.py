from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class TISConfig:
    """Settings of the TIS trials.

    ``K`` cells are drawn from the high and the low pool of a dimension in each of
    ``n_trials`` trials (random generator seeded with ``seed``). Pools are the cells at or
    above the ``q_high`` and at or below the ``q_low`` quantile. ``subsample_eval``
    restricts the query cells to a random subset.
    """

    K: int = 15
    n_trials: int = 400
    seed: int = 42

    # pool construction (quantiles)
    use_quantiles: bool = True
    q_high: float = 0.75
    q_low: float = 0.25

    # compute options
    exclude_query_from_pools: bool = True
    subsample_eval: Optional[int] = None  # evaluate on subset of query cells
    eps: float = 1e-12


def zscore_by_column_dense(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (A - mu) / sd


def l2_normalize_rows_any(X: Union[np.ndarray, sparse.csr_matrix], eps: float = 1e-12):
    """
    Row L2-normalize dense or CSR sparse.
    Returns normalized X and row norms.
    """
    if sparse.issparse(X):
        X = X.tocsr()
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
        norms = np.where(norms < eps, 1.0, norms)
        inv = 1.0 / norms
        Xn = X.multiply(inv[:, None])
        return Xn, norms
    else:
        norms = np.linalg.norm(X, axis=1)
        norms = np.where(norms < eps, 1.0, norms)
        return (X / norms[:, None]).astype(np.float32), norms


def build_pools_quantile(A: np.ndarray, q_low: float, q_high: float) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    For each unit u:
      top_idx[u] = indices with A[:,u] >= q_high quantile
      bot_idx[u] = indices with A[:,u] <= q_low quantile
    Also returns med[u] = median(A[:,u]) for query labeling.
    """
    N, U = A.shape
    top_idx, bot_idx = [], []
    med = np.median(A, axis=0)

    for u in range(U):
        col = A[:, u]
        hi = np.quantile(col, q_high)
        lo = np.quantile(col, q_low)
        top_idx.append(np.where(col >= hi)[0])
        bot_idx.append(np.where(col <= lo)[0])

    return top_idx, bot_idx, med


def average_cosine(Jn: np.ndarray, q_idx: int, set_indices: np.ndarray, eps: float = 1e-12) -> float:
    """
    Cosine similarity between a query row and the L2-normalized centroid of a set of rows.
    Assumes Jn rows are already L2-normalized.
    """
    c = Jn[set_indices].mean(axis=0)
    cn = float(np.linalg.norm(c))
    if cn < eps or not np.isfinite(cn):
        return 0.0
    return float(Jn[q_idx].dot(c / cn))


def compute_tis_mis(
    A: np.ndarray,
    J_like: Union[np.ndarray, sparse.csr_matrix],
    top_idx: List[np.ndarray],
    bot_idx: List[np.ndarray],
    med: np.ndarray,
    cfg: TISConfig,
    *,
    median_from_full_A: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    TIS per dimension. The pool centroid is L2-normalized before the cosine, the query cell
    is excluded from both pools, and ``median_from_full_A`` optionally supplies medians
    computed on the full matrix.
    """
    N, U = A.shape
    rng = np.random.default_rng(cfg.seed)

    # normalize judge space rows
    if sparse.issparse(J_like):
        Jn, _ = l2_normalize_rows_any(J_like, eps=cfg.eps)
        Jn = Jn.toarray().astype(np.float32)  # one-time densification
    else:
        Jn, _ = l2_normalize_rows_any(J_like, eps=cfg.eps)

    MIS = np.full(U, np.nan, dtype=np.float32)

    eval_idx = None
    if cfg.subsample_eval is not None and cfg.subsample_eval < N:
        eval_idx = rng.choice(N, size=int(cfg.subsample_eval), replace=False)

    med_label = median_from_full_A if median_from_full_A is not None else med

    for u in range(U):
        Hpool = np.asarray(top_idx[u])
        Lpool = np.asarray(bot_idx[u])
        if Hpool.size < cfg.K or Lpool.size < cfg.K:
            continue

        correct = 0
        trials_done = 0

        for _ in range(cfg.n_trials):
            q = int(rng.integers(0, N)) if eval_idx is None else int(rng.choice(eval_idx))

            Hp = Hpool
            Lp = Lpool
            if cfg.exclude_query_from_pools:
                if Hp.size:
                    Hp = Hp[Hp != q]
                if Lp.size:
                    Lp = Lp[Lp != q]
                if Hp.size < cfg.K or Lp.size < cfg.K:
                    continue

            H = rng.choice(Hp, size=cfg.K, replace=False)
            L = rng.choice(Lp, size=cfg.K, replace=False)

            sh = average_cosine(Jn, q, H, eps=cfg.eps)
            sl = average_cosine(Jn, q, L, eps=cfg.eps)

            pred_high = (sh > sl)
            true_high = (A[q, u] >= med_label[u])
            correct += int(pred_high == true_high)
            trials_done += 1

        if trials_done > 0:
            MIS[u] = correct / trials_done

    return MIS


def compute_tis_mis_quantile_streamed(
    A: np.ndarray,
    J_like: Union[np.ndarray, sparse.csr_matrix],
    q_low: float,
    q_high: float,
    cfg: TISConfig,
    *,
    median_from_full_A: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Memory-light TIS/MIS computation for large N x U activation matrices.

    This is equivalent to build_pools_quantile(A, q_low, q_high) followed by
    compute_tis_mis(...), but it constructs high/low pools one feature at a
    time instead of storing 2 * U large index arrays simultaneously.
    """
    N, U = A.shape
    rng = np.random.default_rng(cfg.seed)

    if sparse.issparse(J_like):
        Jn, _ = l2_normalize_rows_any(J_like, eps=cfg.eps)
        Jn = Jn.toarray().astype(np.float32)
    else:
        Jn, _ = l2_normalize_rows_any(J_like, eps=cfg.eps)

    MIS = np.full(U, np.nan, dtype=np.float32)

    eval_idx = None
    if cfg.subsample_eval is not None and cfg.subsample_eval < N:
        eval_idx = rng.choice(N, size=int(cfg.subsample_eval), replace=False)

    med = median_from_full_A if median_from_full_A is not None else np.median(A, axis=0)

    for u in range(U):
        col = A[:, u]
        hi = np.quantile(col, q_high)
        lo = np.quantile(col, q_low)
        Hpool = np.where(col >= hi)[0]
        Lpool = np.where(col <= lo)[0]
        if Hpool.size < cfg.K or Lpool.size < cfg.K:
            continue

        correct = 0
        trials_done = 0

        for _ in range(cfg.n_trials):
            q = int(rng.integers(0, N)) if eval_idx is None else int(rng.choice(eval_idx))

            Hp = Hpool
            Lp = Lpool
            if cfg.exclude_query_from_pools:
                Hp = Hp[Hp != q]
                Lp = Lp[Lp != q]
                if Hp.size < cfg.K or Lp.size < cfg.K:
                    continue

            H = rng.choice(Hp, size=cfg.K, replace=False)
            L = rng.choice(Lp, size=cfg.K, replace=False)

            sh = average_cosine(Jn, q, H, eps=cfg.eps)
            sl = average_cosine(Jn, q, L, eps=cfg.eps)

            pred_high = sh > sl
            true_high = A[q, u] >= med[u]
            correct += int(pred_high == true_high)
            trials_done += 1

        if trials_done > 0:
            MIS[u] = correct / trials_done

    return MIS


def shuffle_activations(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffled-activation control: permute the cells independently within each column of A."""
    rng = np.random.default_rng(seed)
    Ash = np.empty_like(A)
    for u in range(A.shape[1]):
        perm = rng.permutation(A.shape[0])
        Ash[:, u] = A[perm, u]
    return Ash


def pca_activations(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    PCA baseline: returns dimension-matched PCA components (same #dims as input).
    """
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    # random_state only matters for the randomized PCA solvers
    pca = PCA(n_components=A.shape[1], random_state=int(seed))
    return pca.fit_transform(A).astype(np.float32)
