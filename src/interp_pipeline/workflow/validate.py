from __future__ import annotations

from typing import Sequence


def validate_parallel_args(
    *,
    labels: Sequence[str],
    min_models: int = 1,
    max_models: int | None = 3,
    **named_lists: Sequence[str] | None,
) -> list[str]:
    """Validate model-parallel CLI arguments.

    Accepts 1, 2 or 3 models by default (``min_models``/``max_models``).
    Every provided list-like argument must have the same length as labels.
    """
    labels = [str(x) for x in labels]
    n = len(labels)

    if n < min_models:
        raise ValueError(f"Need at least {min_models} label(s); got {n}.")
    if max_models is not None and n > max_models:
        raise ValueError(f"Need at most {max_models} label(s); got {n}: {labels}")
    if len(set(labels)) != n:
        raise ValueError(f"Duplicate labels are not allowed: {labels}")

    for name, values in named_lists.items():
        if values is None:
            continue
        values = list(values)
        if len(values) != n:
            raise ValueError(
                f"--{name.replace('_', '-')} must have same length as --labels: "
                f"got {len(values)} vs {n}; labels={labels}"
            )

    return labels
