# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-11-05

First public pre-release of the analysis package.

### Added

- Documentation site (Sphinx, Read the Docs): installation, concepts, data formats, usage, evaluation best practices, roadmap, FAQ and API reference.
- Package metadata (`interpretfm` distribution, version 0.9.0), optional extras `models`, `annotation`, `docs` and `dev`.
- Held-out concept alignment with row and whole-gene splits, including optional de-duplication of cells that appear more than once in an activation store (`dedup_cells`, `--f1-dedup-cells`).
- Example model specification for the all-layer runner (`configs/model_specs.example.json`).
- CPU tests that need no data: imports, command-line help, the quick-start examples on toy data, and held-out F1 regressions.
- Continuous integration (tests and documentation build), pre-commit configuration, contribution guide, code of conduct, security policy, citation file and third-party notices.

[Unreleased]: https://github.com/OliiverHu/interpreTFM/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/OliiverHu/interpreTFM/releases/tag/v0.9.0
