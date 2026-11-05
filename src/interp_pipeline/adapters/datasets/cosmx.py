from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import scanpy as sc

from interp_pipeline.types.dataset import StandardDataset


@dataclass(frozen=True)
class CosMxSpec:
    """
    Minimal dataset spec: the path of an .h5ad file and an optional obs key mapping.
    """
    path: str
    obs_key_map: Optional[Dict[str, str]] = None


class CosMxDatasetAdapter:
    """
    Loads CosMx data from an .h5ad file into a StandardDataset.
    """
    def load(self, cfg: Dict[str, Any] | CosMxSpec) -> StandardDataset:
        if isinstance(cfg, dict):
            spec = CosMxSpec(**cfg)
        else:
            spec = cfg

        adata = sc.read(spec.path)
        ds = StandardDataset(adata=adata, obs_key_map=spec.obs_key_map or {})
        ds.validate()
        return ds
