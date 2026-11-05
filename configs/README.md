# Configuration examples

| File | Used by | Content |
|---|---|---|
| `model_specs.example.json` | `python -m interp_pipeline.layer_experiments.all_layer_sae_f1 --model-spec-json ...` | One entry per model: activation-store root, layers to process, and how cell-level rows are read (`activation_pooling`, `token_value`). The layer lists are the 29 model-layer combinations evaluated in the manuscript. |

Replace every `/path/to/...` with your own location before running a stage.
Stored layer identifiers are zero-based (`layer_1.norm2` is display Layer 2).
See the [usage guide](https://interpretfm.readthedocs.io/en/latest/usage.html) for details.
