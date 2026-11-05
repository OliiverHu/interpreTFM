# Roadmap to v1.0

The 0.9 pre-release makes the analysis package of the manuscript public. Version 1.0, planned for the final publication, turns it into a pipeline that can be applied to a new dataset or model without editing code. Plans can change; the [changelog](changelog.md) records what has been released.

## Interface

- A single command-line entry point that validates a configuration file, runs selected stages and builds a report, with a Python API that mirrors the stages.
- One YAML configuration with dataset, model and evaluation sections; the shipped configurations reproduce the manuscript.
- Cached, resumable stages that each write a manifest (inputs, configuration hash, software versions, run time).

## Stages

1. **Input checks.** Validate the AnnData file, map genes to each model's vocabulary and report coverage, and convert coordinate units to µm.
2. **Extraction** of layer-resolved activations into the sharded store, behind a common adapter interface so that further models can be added.
3. **TIS** per layer with the permutation and PCA controls.
4. **SAE** training with the fixed ℓ1-weight rule, and an untrained SAE of the same width as baseline.
5. **Concept evaluation:** the panel concept universe from g:Profiler, then held-out alignment on the row split and the whole-gene split, against raw dimensions and the untrained SAE, with the per-concept summary.
6. **Cell level:** feature retention and projection; GSI and MHR@100 against raw dimensions and single genes.
7. **Layer summary:** each criterion reported separately, without a composite score.
8. **Spatial** (when coordinates are present): niche sweep with expression-only and label-composition baselines; calibrated interface test with transcript-depth adjustment and spillover checks; ligand–receptor co-localization screen with radii in µm and a matched-neighbor-count option.
9. **Report:** HTML and JSON with every metric next to its baseline, and notes on how to read it.

Datasets without coordinates (single-cell RNA-seq) run all non-spatial stages.

## Distribution

- Per-model conda environments and containers (Docker and Apptainer) for extraction and SAE training.
- A Hugging Face release of trained SAEs and feature–concept tables for the representative layers of each model.
- A tutorial on a small public dataset and an example report.
- Unit tests, a CPU smoke test on tiny data, and reproduction tests against cached activations.
- An archived release with a DOI.

A hosted web service is not planned.
