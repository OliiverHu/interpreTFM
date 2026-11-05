# FAQ

## Does interpreTFM fine-tune the foundation models?

No. Model weights stay frozen. SAEs, niche models and all statistics are fitted downstream of the extracted activations.

## Which models are supported?

scGPT, Geneformer and C2S-Scale. Each has an adapter in `interp_pipeline.adapters.models` that tokenizes an AnnData object and returns per-token hidden states for chosen layers. Another model can be added by writing an adapter that follows `interp_pipeline.adapters.model_base.ModelAdapter`; v1.0 will document this as a supported extension point.

## Do I need spatial data?

No. TIS, SAE training, concept alignment and the cell-level measures need only expression. Niches, interfaces and the co-localization screen need two-dimensional coordinates.

## Do I need a GPU?

For activation extraction and SAE training, in practice yes. TIS, concept alignment and the spatial analyses run on CPU from existing activation stores.

## Why are scores on the row split higher than on the whole-gene split?

They answer different questions. The row split tests new observations of genes whose rows were also used to select features, so a feature that detects a member gene scores well. The whole-gene split tests genes never seen during selection. See [best practices](best_practices.md#hold-out-whole-genes-to-test-concept-generalization).

## Does a high concept F1 mean the model represents the pathway?

Not by itself. It means the feature's activation is associated with membership in an annotated gene set on held-out data. Read it next to raw dimensions and an untrained SAE scored the same way, and next to the whole-gene split.

## Why are concepts restricted to the measured panel?

Targeted spatial panels measure a few hundred genes. Concepts are built from those genes so that every concept is observable in the data; results describe panel-observable concepts, not genome-wide pathways.

## Why does `layer_1.norm2` appear as Layer 2 in figures?

Stored layer identifiers keep the model's zero-based indices; the manuscript numbers layers from 1.

## What does a negative interface z-score mean?

Lower program load on that interface than expected under the spatial null. It does not mean that the cell types avoid each other.

## Importing a module fails with `ModuleNotFoundError: No module named 'plot'`.

Install the repository in editable mode (`python -m pip install -e .`). A few modules import plotting helpers from `plot/` at the repository root.

## Importing a module fails with `ModuleNotFoundError` for `transformers`, `nnsight` or `datasets`.

These are needed only for the model adapters and extraction. Install them with `python -m pip install -e ".[models]"` or use the model's conda environment.

## Where are trained SAEs and processed data?

They are not part of the repository. Trained SAEs and feature–concept tables are planned for a Hugging Face release with v1.0, and processed analysis objects are planned to be archived with the publication.

## How do I report a problem or ask a question?

Open a [GitHub issue](https://github.com/OliiverHu/interpreTFM/issues). Security problems: see `SECURITY.md` in the repository.
