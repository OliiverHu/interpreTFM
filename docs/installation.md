# Installation

## Requirements

- Linux x86-64. The environment files were exported from working Linux installations with CUDA 12.
- Python 3.8 to 3.11. Continuous integration runs the tests on each of these versions; the main environment uses Python 3.8, and the Geneformer and C2S-Scale environments use 3.10 and 3.9.
- An NVIDIA GPU for activation extraction and SAE training. TIS, concept alignment and the spatial analyses run on CPU from existing activation stores, though large stores take time.

## Conda environments (recommended)

scGPT, Geneformer and C2S-Scale depend on conflicting library versions, so each model has its own environment. Activation stores share one format, so all later stages run in the main environment.

| Environment | Use | File |
|---|---|---|
| `interpretfm` | analysis, SAE training and evaluation, scGPT extraction | `envs/interpretfm.yml` |
| `geneformer` | Geneformer extraction | `envs/geneformer.yml` |
| `c2s` | C2S-Scale extraction | `envs/c2s.yml` |

```bash
git clone https://github.com/OliiverHu/interpreTFM.git
cd interpreTFM
conda env create -f envs/interpretfm.yml
conda activate interpretfm
python -m pip install -e .
```

Repeat with `envs/geneformer.yml` or `envs/c2s.yml` for the other models, and run `python -m pip install -e .` inside each of them.

Notes:

- Install in editable mode (`-e`). Some modules import plotting helpers from `plot/` at the repository root, which a regular (non-editable) install does not contain.
- The Geneformer package is not on PyPI. In the `geneformer` environment, install it from its [Hugging Face repository](https://huggingface.co/ctheodoris/Geneformer) (`git lfs install`, clone the repository, then `python -m pip install .` inside it).
- `flash-attn` is commented out in `envs/interpretfm.yml` because it needs a CUDA toolchain to build; interpreTFM does not import it, and the line can be uncommented where a CUDA toolchain is available.
- `python -m pip install -e .` only adds packages that are missing; it does not change versions that the environment already provides.

## pip only

Install PyTorch for your platform first (see [pytorch.org](https://pytorch.org)), then:

```bash
python -m pip install -e ".[models]"      # adds transformers, nnsight and datasets
```

Optional extras: `annotation` (GO hierarchy tools used by `interp_pipeline.downstream.f1.go_reduce`), `docs` (documentation build) and `dev` (tests and linting).

## Model weights and data

Weights, datasets and activation stores are not distributed with the code.

| Model | Source | Notes |
|---|---|---|
| scGPT | [scGPT repository](https://github.com/bowang-lab/scGPT), whole-human checkpoint | the checkpoint folder must contain `best_model.pt` and `vocab.json` |
| Geneformer | [Hugging Face: ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) | model folder; token dictionaries are read from the installed Geneformer package |
| C2S-Scale | [Hugging Face: vandijklab](https://huggingface.co/vandijklab) | model name or local path |

Check each model's license before use. See [data and configuration formats](data.md) for the expected AnnData fields.

## Check the installation

```bash
python -c "import interp_pipeline; print(interp_pipeline.__version__)"
python -m pip install -e ".[dev]"
python -m pytest
```

The tests run on CPU without data or model weights. Tests that need a real activation store are skipped with a message that names the environment variable to set.

## Build the documentation

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

The API reference is generated from the source tree with heavy dependencies mocked, so the package does not need to be installed for this step. `envs/docs.yml` is the environment used to build these pages.
