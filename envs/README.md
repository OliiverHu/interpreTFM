# Conda environments

scGPT, Geneformer and C2S-Scale need conflicting library versions, so each model has its own environment. Activation stores share one format, so all stages after extraction run in the main environment.

| Environment | Use | File |
|---|---|---|
| `interpretfm` | analysis, SAE training and evaluation, scGPT extraction | `interpretfm.yml` |
| `geneformer` | Geneformer extraction | `geneformer.yml` |
| `c2s` | C2S-Scale extraction | `c2s.yml` |
| `interpretfm-docs` | building the documentation | `docs.yml` |

```bash
conda env create -f envs/interpretfm.yml
conda activate interpretfm
python -m pip install -e .
```

Notes:

- The model files are exports of working Linux x86-64 installations with CUDA 12 (`conda env export`, machine-specific lines removed). Pip-only packages are listed under `pip:`.
- Install this repository with `python -m pip install -e .` inside each environment after creating it.
- `geneformer.yml`: the Geneformer package is not on PyPI; install it from <https://huggingface.co/ctheodoris/Geneformer>. PyTorch comes from the PyTorch CUDA 12.1 index (`--extra-index-url` in the file).
- `interpretfm.yml` lists `flash-attn` as a commented-out line, because it needs a CUDA toolchain to build; interpreTFM does not import it, and the line can be uncommented where a CUDA toolchain is available.
- Model weights are not part of any environment; see the installation guide.
