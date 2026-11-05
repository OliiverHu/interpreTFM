# Data and configuration formats

## AnnData input

interpreTFM reads one `.h5ad` file per section (or per single-cell dataset). Cells and model representations are never pooled across datasets.

| Field | Requirement |
|---|---|
| `X` | Raw transcript counts, cells × genes. Each model's tokenization applies its own normalization: scGPT log-transforms nonzero counts, Geneformer ranks genes from raw counts, and C2S-Scale normalizes and log-transforms before ranking genes. |
| `obs_names` | Stable, unique cell identifiers. They become the `example_ids` of the activation store and are used to align activations back to cells. |
| `var` gene symbols | HGNC symbols in `var["feature_name"]` (or `var["index"]`). scGPT maps symbols to its vocabulary; C2S-Scale writes symbols into cell sentences. |
| `var["ensembl_id"]` | Ensembl gene IDs, needed by Geneformer (taken from `var_names` if the column is missing). |
| `obs` cell labels | Optional. Used only downstream, for calibration, composition summaries and interfaces, never to train SAEs or select features. |
| `obsm["spatial"]` | Two-dimensional coordinates, for the spatial stages only. |

### Gene symbols

- Use one symbol convention for the whole analysis. Concept membership, model vocabularies and marker sets are all looked up by gene identifier; a gene whose symbol differs between sources silently drops out.
- Keep the mapping between symbols and Ensembl IDs in `var`. The concept-alignment evaluator uses it to match the token identifiers in an activation store (Ensembl IDs, symbols, or scGPT vocabulary ids) to the rows of the concept matrix.
- Before extraction, remove genes outside the compared models' vocabularies (the scGPT tokenizer warns about unknown symbols but does not drop them) and cells with no counts, and record how many genes and cells were removed.

### Spatial coordinates and units

Coordinates are used in their stored unit, so a radius means different physical distances on different platforms. For example, CosMx coordinates are often pixels of 0.18 µm, while Xenium coordinates are in micrometers: a radius of 100 units is about 18 µm in the first and 100 µm in the second.

- Record the micrometers per coordinate unit for every dataset.
- Give every radius, tile and block size both in coordinate units and in µm.
- When comparing platforms, also compare the mean number of neighbors within the radius, not only the distance.

## Activation stores

Extraction writes one store per model and dataset:

```text
<store_root>/
  activations/
    <layer>/                    e.g. layer_1.norm2 (scGPT), layer_1 (Geneformer, C2S-Scale)
      shard_0/
        activations.pt          tensor, rows x hidden dimension
        index.pt                {"example_ids": [...], "token_ids": [...], "token_unit": "gene"}
        metadata.json           layer, shard, number of rows, hidden dimension, dtype
      shard_1/
      ...
```

Each row of `activations.pt` is one token of one cell. `index.pt` gives, for every row, the cell identifier (`example_ids`) and the token identifier (`token_ids`: an Ensembl ID, a gene symbol, a vocabulary id, or a special token such as the classification token). If a store holds a cell more than once, the concept-alignment evaluator can keep only the first copy (`dedup_cells=True`).

Layer identifiers are the model's zero-based indices. Figures and text in the manuscript number layers from 1.

## Concept matrix

A CSV file with genes as rows (index column: the gene identifier used in the store, usually the Ensembl ID or symbol) and concepts as columns (for example `GO:0006955`), with values 0 or 1. Build it from the measured panel only (see [concepts](concepts.md#concept-universe)).

## SAE checkpoints

A `torch.save` dictionary:

```python
{"state_dict": {...}, "d_in": hidden_dim, "n_latents": number_of_features}
```

The evaluator reads the encoder weight from `encoder.weight` (or `W_enc`), an optional encoder bias from `encoder.bias` (or `b_enc`) and an optional input bias from `bias` (or `x_bias`). Features are `ReLU(W_enc (x - input_bias) + encoder_bias)`. Any representation that can be written in this form can be scored the same way, including raw hidden dimensions and untrained SAEs ([best practices](best_practices.md)).

## Model specification for the all-layer runner

`configs/model_specs.example.json` lists, for each model, the activation-store root, the layers to process and how cell-level rows are read. Pass it with `--model-spec-json`:

```json
{
  "scgpt": {
    "store_root": "/path/to/activation_stores/scgpt",
    "activation_pooling": "token",
    "token_value": "60695",
    "cell_feature_pooling": "mean",
    "layers": ["layer_0.norm2", "layer_1.norm2"]
  }
}
```

- `activation_pooling`: `"token"` reads one row per cell, identified by `token_value` (the scGPT classification-token id `60695`, or `<cls>` for Geneformer); `"mean"` averages a cell's rows (C2S-Scale).
- `cell_feature_pooling`: how gene-token features are pooled into cell-level features downstream.

## Cell-group and interaction tables

The interface functions in `interp_pipeline.downstream.interaction` read two small CSV files:

- a cell-type-to-group mapping with columns `cell_type` and `group`;
- target interaction sets with columns `interaction_name`, `side_a_groups` and `side_b_groups` (semicolon-separated group names).
