from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from tqdm import tqdm

from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.types.dataset import StandardDataset


def extract_c2s_shard(
    adata,
    model_path: str,
    output_dir: str,
    layers: List[str],
    shard: int,
    batch_size: int = 8,
    max_genes: int = 256,
    device: str = "cuda:0",
    pooling: str = "last",
    save_dtype: str = "fp16",
    pool_dtype: str = "fp32",
    normalize: bool = True,
    cache_dir: str | None = None,
    celltype_col: str | None = None,
) -> None:
    """
    Run C2S extraction for one shard AnnData subset.
    """
    output_path = Path(output_dir)
    store = ActivationStore(ActivationStoreSpec(root=str(output_path)))
    dataset = StandardDataset(adata=adata, obs_key_map={})

    spec = ModelSpec(
        name="c2s-scale",
        checkpoint=model_path,
        device=device,
        options={
            "max_genes": max_genes,
            "cache_dir": cache_dir,
            "celltype_col": celltype_col,
        },
    )

    adapter = C2SScaleAdapter()
    handle = adapter.load(spec)

    (output_path / "metadata").mkdir(parents=True, exist_ok=True)
    with open(output_path / "metadata" / f"shard_{shard}.json", "w") as f:
        json.dump(
            {
                "model": model_path,
                "layers": layers,
                "shard": shard,
                "batch_size": batch_size,
                "max_genes": max_genes,
                "pooling": pooling,
                "save_dtype": save_dtype,
                "pool_dtype": pool_dtype,
                "normalize": normalize,
            },
            f,
            indent=2,
        )

    batch_iter = adapter.make_batches(
        dataset=dataset,
        model_handle=handle,
        batch_size=batch_size,
        max_genes=max_genes,
        normalize=normalize,
        celltype_col=celltype_col,
    )

    desc = f"extract:c2sscale:shard_{shard}"
    total_batches = None
    try:
        total_batches = len(batch_iter)
    except Exception:
        total_batches = None

    layer_acts_parts: Dict[str, List[torch.Tensor]] = {layer_name: [] for layer_name in layers}
    layer_example_ids: Dict[str, List[str]] = {layer_name: [] for layer_name in layers}
    layer_token_ids: Dict[str, List[str]] = {layer_name: [] for layer_name in layers}

    for _batch_idx, batch in enumerate(tqdm(batch_iter, total=total_batches, desc=desc)):
        batch["pooling"] = pooling
        batch["save_dtype"] = save_dtype
        batch["pool_dtype"] = pool_dtype

        captured = adapter.forward_and_capture(
            model_handle=handle,
            batch=batch,
            layers=layers,
            capture_cfg={},
        )

        processed = adapter.process_captured(
            captured=captured,
            batch=batch,
        )

        for layer_name, out in processed.items():
            layer_acts_parts[layer_name].append(out["acts"])
            layer_example_ids[layer_name].extend([str(x) for x in out["ex"]])
            layer_token_ids[layer_name].extend([str(x) for x in out["tok"]])

    for layer_name in layers:
        acts_parts = layer_acts_parts[layer_name]
        if not acts_parts:
            continue
        acts_cat = torch.cat(acts_parts, dim=0)
        store.write_token_shard(
            layer=layer_name,
            shard_id=int(shard),
            acts=acts_cat,
            example_ids=layer_example_ids[layer_name],
            token_ids=layer_token_ids[layer_name],
            token_unit="gene",
            meta={
                "source_format": "c2s_shard_compacted",
                "source_layer": layer_name,
                "source_batches": len(acts_parts),
            },
        )


def extract_c2s_dataset(
    adata,
    model_path: str,
    output_dir: str,
    layers: List[str],
    shards: int = 60,
    shard_key: str = "shards",
    batch_size: int = 8,
    max_genes: int = 256,
    device: str = "cuda:0",
    pooling: str = "last",
    save_dtype: str = "fp16",
    pool_dtype: str = "fp32",
    normalize: bool = True,
    cache_dir: str | None = None,
    celltype_col: str | None = None,
) -> None:
    """
    Run C2S extraction over all shard subsets in an AnnData.
    """
    print(f"Loaded adata: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Using layers: {layers}")
    print(
        f"shards={shards}, batch_size={batch_size}, max_genes={max_genes}, "
        f"pooling={pooling}, save_dtype={save_dtype}, pool_dtype={pool_dtype}"
    )

    for s in range(shards):
        subset = adata[adata.obs[shard_key] == f"shard_{s}"].copy()
        print(f"\n[c2s_scale] processing shard_{s}: {subset.n_obs} cells")

        if subset.n_obs == 0:
            continue

        extract_c2s_shard(
            adata=subset,
            model_path=model_path,
            output_dir=output_dir,
            layers=layers,
            shard=s,
            batch_size=batch_size,
            max_genes=max_genes,
            device=device,
            pooling=pooling,
            save_dtype=save_dtype,
            pool_dtype=pool_dtype,
            normalize=normalize,
            cache_dir=cache_dir,
            celltype_col=celltype_col,
        )
