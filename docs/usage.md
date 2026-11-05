# Usage

In the 0.9 pre-release, stages are run as Python modules (`python -m ...`) or called from Python. A single command that runs every stage from one configuration file is planned for v1.0 ([roadmap](roadmap.md)). The job-planning commands (section 2), the TIS and concept-alignment snippets (sections 3 and 4) and the niche sweep (section 6) are exercised on toy data by `tests/test_smoke.py`.

## Stages at a glance

| Stage | Module | How to run in 0.9 |
|---|---|---|
| Activation extraction | `interp_pipeline.extraction` | Python, in the model's environment (GPU) |
| Concept universe | `interp_pipeline.get_annotation.gprofiler_client` | Python (queries); writing the gene × concept matrix: v1.0 |
| TIS | `interp_pipeline.tis.core`, `interp_pipeline.tis.io` | Python |
| TIS layer screen | `interp_pipeline.tis.single_layer_screen` | needs a backend module not included (v1.0): `--backend-script` has no default |
| SAE training and QC | `interp_pipeline.layer_experiments.all_layer_sae_f1` (`--stages sae qc`) | needs trainer and QC modules not included (v1.0): `--train-script` and `--qc-script` have no default; `interp_pipeline.sae.trainers` fits a bias-free SAE from Python |
| Held-out concept alignment | `interp_pipeline.get_annotation.f1_alignment` | Python, or `all_layer_sae_f1 --stages f1` |
| Result aggregation | `interp_pipeline.layer_experiments.aggregate_sae_f1` | `python -m` |
| Niche discovery | `interp_pipeline.downstream.context` | Python |
| Interface scoring | `interp_pipeline.downstream.interaction` | Python |

Write outputs to a directory outside the source tree (for example `runs/`, which is ignored by git).

## 1. Extract activations

Extraction runs in the environment of the model (see [installation](installation.md)) and writes a sharded [activation store](data.md#activation-stores). The functions are `interp_pipeline.extraction.scgpt_extraction.extract_activations` (with the scGPT adapter), `interp_pipeline.extraction.geneformer_extraction.prepare_geneformer_h5ad`, `tokenize_geneformer_dataset` and `extract_geneformer_to_store`, and `interp_pipeline.extraction.c2s_extraction.extract_c2s_dataset` (which expects cells pre-assigned to shards in an `obs` column). See the [API reference](api/extraction.md) for their arguments.

## 2. Plan the all-layer jobs

`all_layer_sae_f1` runs SAE training, SAE quality control and held-out concept alignment for every model and layer listed in a model specification. Start from the example and replace the store paths:

```bash
cp configs/model_specs.example.json my_model_specs.json   # edit store_root for each model

python -m interp_pipeline.layer_experiments.all_layer_sae_f1 --help
python -m interp_pipeline.layer_experiments.all_layer_sae_f1 \
    --model-spec-json my_model_specs.json \
    --out-root runs/all_layers --status-only
```

`--status-only` writes `runs/all_layers/manifest.json` and `status_<run_tag>.csv` (one row per model, layer and stage) and runs nothing; `--dry-run` additionally lists every job. `--models scgpt` restricts the run to one model. Outputs are organized as `<out-root>/<model>/<layer_tag>/{sae,activation_qc,f1_heldout}/<run_tag>/`, where `<layer_tag>` is the layer name made path-safe by replacing `.` and `/` with `_` (`layer_0.norm2` → `layer_0_norm2`). Finished stages are skipped on the next run unless `--force` is given.

The `sae` and `qc` stages load external trainer and quality-control modules given by `--train-script` and `--qc-script`. These modules are not part of 0.9 and the options have no default: without them, a run that includes these stages stops with a message. Run `--stages f1` with SAE checkpoints you already have.

The `f1` stage expects the SAE checkpoint of each job at `<out-root>/<model>/<layer_tag>/sae/<run_tag>/sae_<layer>_best.pt` (`sae_<layer_tag>_best.pt` is also found) and needs `--gt-csv` (the [concept matrix](data.md#concept-matrix)) and `--adata-path`. For scGPT stores it also needs the scGPT checkpoint folder (`--scgpt-foundation-ckpt`) to decode vocabulary ids. By default it first writes a copy of the AnnData with standardized gene-identifier columns to `<out-root>/_tmp/` (`--no-patch-adata-schema` turns this off).

## 3. TIS

```python
import anndata as ad
import numpy as np
from interp_pipeline.tis.core import TISConfig, compute_tis_mis_quantile_streamed, shuffle_activations
from interp_pipeline.tis.io import build_judge_matrix, extract_cls_from_shards_aligned

adata = ad.read_h5ad("/path/to/section.h5ad")
A, has_row = extract_cls_from_shards_aligned(
    "/path/to/activation_stores/scgpt/activations/layer_1.norm2", obs_names=adata.obs_names
)
J = build_judge_matrix(adata[has_row], mode="log1p_cp10k")   # similarity is computed on expression
cfg = TISConfig(K=32, n_trials=800, seed=42)                  # 32 reference cells per pool, 800 trials
tis = compute_tis_mis_quantile_streamed(A[has_row], J, q_low=0.25, q_high=0.75, cfg=cfg)
tis_null = compute_tis_mis_quantile_streamed(shuffle_activations(A[has_row]), J, 0.25, 0.75, cfg)
gain = np.nanmean(tis) - np.nanmean(tis_null)              # layer-level TIS gain
```

`extract_cls_from_shards_aligned` reads one row per cell (the scGPT classification token by default; pass `cls_token_id` for other ids). `pca_activations` in the same module gives the dimension-matched PCA reference.

## 4. Held-out concept alignment

```python
from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer

for unit in ("gene_token_row", "gene_identity"):        # row split and whole-gene split
    heldout_report_for_layer(
        layer="layer_1.norm2",
        store_root="/path/to/activation_stores/scgpt",
        gt_csv="/path/to/gene_by_concept.csv",
        sae_ckpt_path="/path/to/sae.pt",
        out_dir=f"runs/f1/scgpt/layer_1.norm2/{unit}",
        latent_thresholds=[0.0, 0.15, 0.3, 0.6],
        valid_frac=0.2,
        test_frac=0.1,
        seed=0,
        heldout_unit=unit,
        adata_path="/path/to/section.h5ad",           # symbol / Ensembl mapping
        scgpt_ckpt="/path/to/scGPT_human",            # scGPT only: decodes vocabulary ids
        device="cuda",
    )
```

Each call writes `valid_concept_f1_scores.csv` (top features per concept and threshold on the validation partition), `test_concept_f1_scores.csv` (those features scored on the test partition), `selected_features_valid.json`, `concept_support_by_split.csv`, `heldout_gene_split.json` and `counts_summary.json` (rows read, mapped and assigned to each partition). Run the same call with raw-dimension and untrained-SAE checkpoints for the baselines ([best practices](best_practices.md#report-every-feature-metric-next-to-its-baselines)). For activation stores in which a cell appears more than once, pass `dedup_cells=True` to keep only its first copy.

## 5. Aggregate per-layer results

```bash
python -m interp_pipeline.layer_experiments.aggregate_sae_f1 --root runs/all_layers --tis-root runs/tis
```

This collects the SAE, QC and F1 summaries under `--root` into `all_layer_sae_f1_summary.csv`, merges per-layer TIS summaries if `--tis-root` is given, and plots one metric per model (`--metric`).

## 6. Niches

```python
from interp_pipeline.downstream.context.sweep import SweepConfig, run_niche_sweep

cfg = SweepConfig(
    radii=[100.0, 120.0, 150.0],          # coordinate units: record the µm equivalent
    n_clusters_list=[3, 4, 5],
    method_list=["kmeans", "gmm"],
    space_list=["m", "xm"],               # microenvironment only, or cell + microenvironment
    seeds=[0, 1, 2],
    min_cluster_size=150,
)
table = run_niche_sweep(X, coords, cfg)   # X: cells x features, coords: cells x 2
```

The table has one row per configuration with seed-to-seed ARI, neighborhood purity and its tile-shuffle z-score, centroid separability, fragmentation and the ranking `score`. Run the same sweep on components of log-normalized expression for the expression-only baseline. `interp_pipeline.downstream.context.validate` has the cell-type contingency residuals and cross-model niche matching.

## 7. Interfaces

`interp_pipeline.downstream.interaction.core` provides the building blocks of the interface analysis: cell program scores (`cell_program_scores`), radius edges (`build_edges`), edge scores, group-pair tables per niche (`domain_edge_masks`, `group_pair_keys_for_edges`, `build_metric_table`) and a within-tile permutation null (`make_tiles`, `compute_null_stats`). The transcript-count adjustment, the stratified null, the contact-free calibration of the call threshold and the spillover checks described in [concepts](concepts.md#interface-statistic-and-spillover-checks) are planned for v1.0.
