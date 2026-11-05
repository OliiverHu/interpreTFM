from __future__ import annotations

from typing import List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


def pair_key(a: str, b: str) -> str:
    a, b = str(a), str(b)
    return f"{a}||{b}" if a <= b else f"{b}||{a}"


def split_pair_key(k: str) -> Tuple[str, str]:
    return tuple(str(k).split("||", 1))  # type: ignore


def split_semicolon(x: str) -> List[str]:
    return [z.strip() for z in str(x).split(";") if z.strip()]


def load_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"cell_type", "group"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"celltype group CSV missing columns: {miss}")
    df["cell_type"] = df["cell_type"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def load_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"interaction_name", "side_a_groups", "side_b_groups"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"target interaction CSV missing columns: {miss}")
    for c in need:
        df[c] = df[c].astype(str)
    return df


def load_X(a: ad.AnnData) -> np.ndarray:
    X = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    X = X.astype(np.float32, copy=False)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def cell_program_scores(X: np.ndarray, clip_quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell program scores from a cells x features matrix.

    Returns ``(raw, norm)``: ``raw`` is the mean of the positive part of each row; ``norm``
    clips each feature at its ``clip_quantile`` quantile, z-scores it across cells and
    averages the positive z-scores of each row.
    """
    Xp = np.maximum(X, 0.0)
    raw = Xp.mean(axis=1).astype(np.float32)
    Xc = Xp.copy()
    if 0.0 < clip_quantile < 1.0:
        hi = np.quantile(Xc, clip_quantile, axis=0).astype(np.float32)
        Xc = np.minimum(Xc, hi[None, :])
    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (Xc - mu[None, :]) / sd[None, :]
    norm = np.maximum(Z, 0.0).mean(axis=1).astype(np.float32)
    return raw, norm


def build_edges(coords: np.ndarray, radius: float) -> np.ndarray:
    """All cell pairs closer than ``radius``, as an (n_edges, 2) array of cell indices."""
    tree = cKDTree(coords)
    try:
        pairs = tree.query_pairs(float(radius), output_type="ndarray")
    except TypeError:
        pairs = np.array(list(tree.query_pairs(float(radius))), dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def make_tiles(coords: np.ndarray, tile_size: float) -> List[np.ndarray]:
    """Cell indices of each square spatial tile of side ``tile_size`` (used by the permutation null)."""
    xy = np.floor(coords / float(tile_size)).astype(np.int64)
    tile_id = np.array([f"{x}_{y}" for x, y in xy], dtype=object)
    return [np.where(tile_id == t)[0] for t in pd.unique(tile_id)]


def shuffle_within_tiles(values: np.ndarray, tile_indices: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for idx in tile_indices:
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def edge_score(edges: np.ndarray, cell_score: np.ndarray) -> np.ndarray:
    return (0.5 * (cell_score[edges[:, 0]] + cell_score[edges[:, 1]])).astype(np.float32)


def domain_edge_masks(edges: np.ndarray, niche: np.ndarray) -> dict:
    """Boolean edge masks: ``"global"`` (all edges) and ``"niche<v>"`` (both cells in niche v)."""
    niche = pd.Series(niche).astype(str).to_numpy()
    out = {"global": np.ones(len(edges), dtype=bool)}
    for v in sorted(pd.unique(niche)):
        cm = niche == str(v)
        out[f"niche{v}"] = cm[edges[:, 0]] & cm[edges[:, 1]]
    return out


def group_pair_keys_for_edges(edges: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Unordered group-pair key ("a||b") of every edge, from the group label of each cell."""
    return np.array([pair_key(groups[i], groups[j]) for i, j in edges], dtype=object)


def build_metric_table(group_pair_keys: np.ndarray, domain_masks: dict, domains: Sequence[str], min_edges: int) -> pd.DataFrame:
    """One row per domain and group pair with at least ``min_edges`` edges (columns domain, group_pair_key, group_a, group_b, n_edges)."""
    rows = []
    for dom in domains:
        if dom not in domain_masks:
            continue
        dm = domain_masks[dom]
        vc = pd.Series(group_pair_keys[dm]).value_counts()
        for gkey, n in vc.items():
            if int(n) < int(min_edges):
                continue
            a, b = split_pair_key(gkey)
            rows.append({"domain": dom, "group_pair_key": gkey, "group_a": a, "group_b": b, "n_edges": int(n)})
    return pd.DataFrame(rows)


def target_axis_for_pair(gkey: str, targets: pd.DataFrame) -> str:
    ga, gb = split_pair_key(gkey)
    hits = []
    for _, r in targets.iterrows():
        A = set(split_semicolon(r["side_a_groups"]))
        B = set(split_semicolon(r["side_b_groups"]))
        if (ga in A and gb in B) or (gb in A and ga in B):
            hits.append(str(r["interaction_name"]))
    return ";".join(hits)


def compute_null_stats(
    metric_df: pd.DataFrame,
    edges: np.ndarray,
    group_pair_keys: np.ndarray,
    domain_masks: dict,
    obs_edge_norm: np.ndarray,
    obs_edge_raw: np.ndarray,
    cell_norm: np.ndarray,
    cell_raw: np.ndarray,
    tile_indices: Sequence[np.ndarray],
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    """Observed edge scores of each row of ``metric_df`` and their within-tile permutation null.

    For every domain and group pair, the mean edge score (normalized and raw program scores)
    is compared with ``n_perm`` permutations in which cell scores are shuffled within spatial
    tiles. Adds the observed scores, null mean and SD, z-score and upper empirical p-value
    for both scores, and a z-score of the observed normalized score against all edges.
    """
    rng = np.random.default_rng(seed)
    masks = []
    for _, r in metric_df.iterrows():
        m = domain_masks[str(r["domain"])] & (group_pair_keys == str(r["group_pair_key"]))
        masks.append(np.where(m)[0])

    out = metric_df.copy()
    obs_norm = np.array([float(obs_edge_norm[idx].mean()) if len(idx) else np.nan for idx in masks])
    obs_raw = np.array([float(obs_edge_raw[idx].mean()) if len(idx) else np.nan for idx in masks])

    edge_mu = float(np.mean(obs_edge_norm))
    edge_sd = float(np.std(obs_edge_norm))
    obs_global_edge_z = np.zeros_like(obs_norm) if edge_sd < 1e-9 else (obs_norm - edge_mu) / edge_sd

    null_norm = np.zeros((len(out), int(n_perm)), dtype=np.float32)
    null_raw = np.zeros((len(out), int(n_perm)), dtype=np.float32)

    for p in range(int(n_perm)):
        sn = shuffle_within_tiles(cell_norm, tile_indices, rng)
        sr = shuffle_within_tiles(cell_raw, tile_indices, rng)
        en = edge_score(edges, sn)
        er = edge_score(edges, sr)
        for k, idx in enumerate(masks):
            null_norm[k, p] = float(en[idx].mean()) if len(idx) else np.nan
            null_raw[k, p] = float(er[idx].mean()) if len(idx) else np.nan

    def fill(prefix: str, obs: np.ndarray, null: np.ndarray):
        mu = np.nanmean(null, axis=1)
        sd = np.nanstd(null, axis=1, ddof=1)
        z = np.where(sd > 1e-9, (obs - mu) / sd, np.nan)
        p = (np.sum(null >= obs[:, None], axis=1) + 1.0) / (null.shape[1] + 1.0)
        out[f"obs_{prefix}_score"] = obs
        out[f"null_{prefix}_mean"] = mu
        out[f"null_{prefix}_sd"] = sd
        out[f"shuffle_{prefix}_z"] = z
        out[f"empirical_p_upper_{prefix}"] = p

    fill("norm", obs_norm, null_norm)
    fill("raw", obs_raw, null_raw)
    out["obs_global_edge_z"] = obs_global_edge_z
    out["n_perm"] = int(n_perm)
    return out
