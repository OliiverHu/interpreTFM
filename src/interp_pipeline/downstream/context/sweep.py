from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .neighbors import NeighborConfig
from .microenv import microenv_embedding
from .cluster import ClusterConfig, build_cluster_matrix, pca_embed, cluster_labels
from .metrics import (
    stability_ari_over_seeds,
    mean_neighbor_purity,
    tile_shuffle_null_purity,
    separability_cosine,
    min_cluster_size,
    fragmentation_score_from_neigh,
)


@dataclass(frozen=True)
class SweepConfig:
    """Grid and settings of the niche sweep (see :func:`run_niche_sweep`).

    Every combination of neighborhood radius, clustering space (``"m"``: microenvironment
    only; ``"xm"``: cell plus microenvironment), method and number of clusters is clustered
    once per seed. ``tile_mult`` x radius is the tile size of the purity null, and the
    ``w_*`` weights combine stability, clipped purity z-score and separability into ``score``.
    """

    radii: List[float]
    n_clusters_list: List[int]
    method_list: List[str]          # ["kmeans","gmm"]
    space_list: List[str]           # ["m","xm"]
    kernel: str = "uniform"         # uniform/gaussian
    sigma_frac: float = 0.25

    pca_components: int = 50
    seeds: List[int] = (0, 1, 2, 3, 4)

    min_cluster_size: int = 150

    # null model
    tile_mult: float = 4.0
    n_null: int = 20
    null_seed: int = 0

    # composite score weights
    w_stability: float = 1.5
    w_purityz: float = 0.1
    w_sep: float = 1.0
    purityz_clip: float = 10.0


def run_niche_sweep(
    X: np.ndarray,
    coords: np.ndarray,
    cfg: SweepConfig,
) -> pd.DataFrame:
    """Cluster cells into niches for every configuration in ``cfg`` and score each configuration.

    ``X`` is a cells x features matrix and ``coords`` the cells' spatial coordinates. Returns
    one row per configuration whose smallest cluster has at least ``cfg.min_cluster_size``
    cells, with seed-to-seed ARI, neighborhood purity and its tile-shuffle z-score,
    separability, fragmentation and the composite ``score``, sorted by ``score``.
    """
    rows: List[Dict] = []

    for r in cfg.radii:
        ncfg = NeighborConfig(radius=float(r), kernel=cfg.kernel, sigma_frac=cfg.sigma_frac)
        m, neigh = microenv_embedding(X, coords, ncfg)

        for space in cfg.space_list:
            Z = build_cluster_matrix(X, m, space=space)
            # Clamp PCA to the actual matrix rank so that inputs with few features
            # do not fail at the sweep stage.
            max_valid_components = int(min(Z.shape[0], Z.shape[1]))
            n_components = int(max(1, min(int(cfg.pca_components), max_valid_components)))
            E0 = pca_embed(Z, n_components=n_components, seed=0)  # fixed PCA seed

            for method in cfg.method_list:
                for k in cfg.n_clusters_list:
                    labels_by_seed = []
                    for sd in cfg.seeds:
                        ccfg = ClusterConfig(
                            space=space,
                            pca_components=n_components,
                            method=method,
                            n_clusters=int(k),
                            seed=int(sd),
                        )
                        # clustering done in PCA space
                        lab = cluster_labels(E0, ccfg)
                        labels_by_seed.append(lab)

                    ari_mean, ari_std = stability_ari_over_seeds(labels_by_seed)
                    labels0 = labels_by_seed[0]
                    stability_term = float(ari_mean) if np.isfinite(ari_mean) else 0.0

                    mcs = min_cluster_size(labels0)
                    if mcs < cfg.min_cluster_size:
                        continue

                    purity = mean_neighbor_purity(labels0, neigh)
                    tile_size = float(cfg.tile_mult) * float(r)
                    null_mu, null_sd = tile_shuffle_null_purity(
                        labels0, coords, neigh, tile_size=tile_size, n_null=cfg.n_null, seed=cfg.null_seed
                    )
                    purity_z = (purity - null_mu) / null_sd if null_sd > 0 else 0.0
                    purity_z_clip = float(np.clip(purity_z, -cfg.purityz_clip, cfg.purityz_clip))

                    sep = separability_cosine(E0, labels0)
                    frag = fragmentation_score_from_neigh(labels0, neigh)

                    score = cfg.w_stability * stability_term + cfg.w_purityz * purity_z_clip + cfg.w_sep * sep

                    rows.append(
                        dict(
                            radius=float(r),
                            space=space,
                            kernel=cfg.kernel,
                            method=method,
                            n_clusters=int(k),
                            n_seeds=int(len(labels_by_seed)),
                            ari_mean=float(ari_mean),
                            ari_std=float(ari_std),
                            min_cluster_size=int(mcs),
                            purity=float(purity),
                            purity_null_mean=float(null_mu),
                            purity_null_std=float(null_sd),
                            purity_z=float(purity_z),
                            separability=float(sep),
                            fragmentation=float(frag),
                            score=float(score),
                        )
                    )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
