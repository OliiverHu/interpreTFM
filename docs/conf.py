"""Sphinx configuration for the interpreTFM documentation."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The API reference is built from the source tree; the package is not installed on Read the Docs.
sys.path.insert(0, str(ROOT / "src"))
# Some modules import the plotting helpers in plot/ at the repository root.
sys.path.insert(0, str(ROOT))

_init = (ROOT / "src" / "interp_pipeline" / "__init__.py").read_text()
release = re.search(r'^__version__ = "([^"]+)"', _init, re.M).group(1)
version = release

project = "interpreTFM"
author = "Yunfei Hu and contributors"
copyright = "2026, Yunfei Hu and contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# --- API reference -------------------------------------------------------------------------
# Heavy or model-specific dependencies are mocked so the reference builds without GPUs,
# model weights or the scientific Python stack.
autodoc_mock_imports = [
    "torch",
    "transformers",
    "nnsight",
    "datasets",
    "geneformer",
    "scgpt",
    "scanpy",
    "anndata",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "tqdm",
    "requests",
    "gprofiler",
    "goatools",
]
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# --- HTML ------------------------------------------------------------------------------------
html_theme = "furo"
html_title = f"interpreTFM {release}"
html_static_path = ["_static"]
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True


# --- Free-form docstrings ----------------------------------------------------------------------
# Many docstrings in the research code are free-form text rather than reStructuredText. Those that
# would not parse cleanly are shown verbatim (as a literal block) instead of being mangled.
from docutils import nodes  # noqa: E402
from docutils.core import publish_doctree  # noqa: E402

_IGNORED = ("Unknown interpreted text role", "Unknown directive type", "No role entry", "No directive entry")


def _parses_cleanly(text: str) -> bool:
    try:
        tree = publish_doctree(text, settings_overrides={"report_level": 5, "halt_level": 5, "warning_stream": False})
    except Exception:
        return False
    for msg in tree.findall(nodes.system_message):
        if msg["level"] >= 2 and not any(s in msg.astext() for s in _IGNORED):
            return False
    return True


def _literal_if_not_rst(app, what, name, obj, options, lines):
    if lines and not _parses_cleanly("\n".join(lines)):
        lines[:] = ["::", ""] + ["    " + line for line in lines] + [""]


def setup(app):
    app.connect("autodoc-process-docstring", _literal_if_not_rst)
