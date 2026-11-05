# interpreTFM

interpreTFM is a layer-resolved framework for evaluating what transcriptomic foundation models encode.
It extracts hidden states from frozen scGPT, Geneformer and C2S-Scale models, decomposes each evaluated layer with a sparse autoencoder (SAE), scores the features against baselines, and projects concept-associated features back to cells for analyses of single-cell and targeted spatial transcriptomic data.

:::{figure} _static/figure1.png
:alt: The interpreTFM workflow (panels a–f) and the downstream interpretability analyses (panels g–i).
:width: 100%

**Figure 1.** The interpreTFM workflow: expression profiles are tokenized for a frozen foundation model (a, b), layer activations are scored with the transcriptomic interpretability score (c) and decomposed by a sparse autoencoder (d), features are evaluated by held-out concept alignment and cell-level specificity (e) and projected to cells (f), and the resulting profiles support layer-wise comparison, niche discovery and interface analysis (g–i).
:::

## What interpreTFM measures

- **Transcriptomic interpretability score (TIS):** whether variation along a representation dimension is reflected in measured expression.
- **Held-out concept alignment:** whether SAE features mark panel-observable biological concepts on held-out gene-token rows or held-out whole genes, compared with raw hidden dimensions and an untrained SAE of the same width.
- **Population specificity (GSI, MHR@100):** how specific cell-level features are to cell populations, compared with single genes.
- **Spatial niches:** niches built from the features, compared with expression-only and label-composition baselines.
- **Interfaces:** program enrichment on cell-group interfaces under a calibrated statistic, with checks for transcript spillover.
- **Ligand–receptor co-localization:** a model-free screen that places candidate pairs in tissue context.

Each measure is meant to be read next to its baseline. The [concepts page](concepts.md) explains them in plain language, and the [best practices](best_practices.md) list how to report them.

:::{note}
This documentation describes the 0.9 pre-release. It contains the `interp_pipeline` package used for the manuscript's analyses; several steps still run from drivers that are not part of the pre-release and are marked as such on the [concepts page](concepts.md).
A v1.0 pipeline with a single entry point, per-model containers, trained SAEs on Hugging Face and a tutorial is in preparation ([roadmap](roadmap.md)).
:::

```{toctree}
:maxdepth: 2
:caption: User guide

installation
concepts
data
usage
best_practices
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
faq
roadmap
changelog
citation
```
