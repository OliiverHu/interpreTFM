# Contributing to interpreTFM

Thank you for your interest in interpreTFM. Bug reports, questions, documentation fixes and code contributions are welcome.

interpreTFM is research software. The 0.9 package is kept close to the code used for the manuscript, and a v1.0 pipeline is being built ([roadmap](https://interpretfm.readthedocs.io/en/latest/roadmap.html)). For anything beyond a small fix, please open an issue first so that we can agree on the approach before you invest time.

## Reporting a problem

Open a [GitHub issue](https://github.com/OliiverHu/interpreTFM/issues) and include:

- what you ran (command or code), what you expected and what happened, with the full error message;
- the interpreTFM version (`python -c "import interp_pipeline; print(interp_pipeline.__version__)"`), Python version, operating system, and the environment file you used;
- the model and the shape of your data (number of cells and genes, platform). Please do not attach patient-level or otherwise restricted data.

Security problems: see [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/OliiverHu/interpreTFM.git
cd interpreTFM
conda env create -f envs/interpretfm.yml      # or any environment with PyTorch installed
conda activate interpretfm
python -m pip install -e ".[dev]"
pre-commit install
```

## Tests

```bash
python -m pytest
```

Tests must run on CPU without private data or model weights. A test that needs a real dataset or activation store must skip with a clear reason when the data are absent, and take the location from an environment variable (for example with `pytest.mark.skipif`).

## Code style

- `ruff check .` runs a deliberately small rule set (syntax errors and undefined names; see `pyproject.toml`). Please do not reformat existing modules in the same pull request as a functional change.
- New code: type hints, docstrings in Google or NumPy style, no hard-coded paths, and no network access or file writes at import time.
- Changes to evaluation logic must keep the published settings reproducible: add an option rather than changing a default, and add a test that pins the old behavior.

## Documentation

The documentation lives in `docs/` (Markdown with MyST, built with Sphinx):

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Pull requests

1. Fork the repository and create a branch from the default branch.
2. Make focused commits with clear messages.
3. Run the tests and build the documentation.
4. Update `CHANGELOG.md` under "Unreleased".
5. Open the pull request and describe what changed and why.

By contributing, you agree that your contributions are licensed under the GNU General Public License v3.0 or later, the license of this project.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.
