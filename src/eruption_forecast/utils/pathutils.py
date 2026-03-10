"""File path resolution utilities.

This module provides utilities for resolving output directory paths relative
to a root anchor directory, creating directories, and loading JSON files.
"""

import os
import json
from typing import Any, Literal


def ensure_dir(path: str) -> str:
    """Create a directory (and any missing parents) if it does not already exist.

    A thin, named wrapper around ``os.makedirs(path, exist_ok=True)`` that
    returns the path so callers can chain it inline.

    Args:
        path (str): Directory path to create.

    Returns:
        str: The same ``path`` that was passed in.
    """
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> Any:
    """Load and parse a JSON file.

    Opens ``path`` in text mode, parses its contents with ``json.load``, and
    returns the resulting Python object.  Raises ``FileNotFoundError`` if the
    file is absent so callers get a clear error instead of a generic
    ``OSError``.

    Args:
        path (str): Path to the JSON file to read.

    Returns:
        Any: Parsed JSON content (dict, list, str, int, float, bool, or None).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def resolve_output_dir(
    output_dir: str | None,
    root_dir: str | None,
    default_subpath: str,
) -> str:
    """Resolve an output directory path against an anchor directory.

    Provides a consistent way to resolve output paths relative to a stable
    root directory instead of relying on the current working directory. This
    is critical for the pipeline's output directory structure.

    Resolution rules:
    1. Absolute ``output_dir`` → used as-is (``root_dir`` is ignored).
    2. Relative ``output_dir`` → joined with ``root_dir`` (or ``os.getcwd()`` if None).
    3. ``None`` ``output_dir`` → ``root_dir / default_subpath``.

    Args:
        output_dir (str | None): Caller-supplied output directory (absolute, relative, or None).
        root_dir (str | None): Anchor directory for resolving relative paths.
            If None, falls back to ``os.getcwd()``.
        default_subpath (str): Sub-path appended to the anchor when ``output_dir`` is None.

    Returns:
        str: Resolved absolute or anchored output directory path.

    Examples:
        >>> resolve_output_dir(None, "/data/project", "output")
        '/data/project/output'
        >>> resolve_output_dir("custom", "/data/project", "output")
        '/data/project/custom'
        >>> resolve_output_dir("/abs/path", "/data/project", "output")
        '/abs/path'
    """
    anchor = root_dir if root_dir is not None else os.getcwd()
    if output_dir is None:
        return os.path.join(anchor, default_subpath)
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(anchor, output_dir)


def build_model_directories(
    root_dir: str,
    classifier_slug: str,
    cv_slug: str,
    mode: Literal["with-evaluation", "only"],
) -> dict[str, str]:
    """Build standardized model output directory structure.

    Creates a hierarchical directory structure for model training outputs under
    ``trainings/{mode}/{classifier_slug}/{cv_slug}/``.

    Args:
        root_dir (str): Base output directory path.
        classifier_slug (str): Slugified classifier name (e.g., ``"xgb-classifier"``).
        cv_slug (str): Slugified CV strategy name (e.g., ``"stratified-shuffle-split"``).
        mode (Literal["with-evaluation", "only"]): Training mode —
            ``"with-evaluation"`` adds ``metrics/`` and ``figures/`` subdirectories.

    Returns:
        dict[str, str]: Dictionary with directory paths:
            - ``"base"`` (str): Top-level training directory.
            - ``"features"`` (str): Features output directory.
            - ``"significant_features"`` (str): Significant features subdirectory.
            - ``"models"`` (str): Trained models directory.
            - ``"metrics"`` (str): Evaluation metrics directory (only when
              ``mode="with-evaluation"``).
            - ``"figures"`` (str): Plot figures directory (only when
              ``mode="with-evaluation"``).

    Examples:
        >>> dirs = build_model_directories(
        ...     root_dir="/path/to/output",
        ...     classifier_slug="random-forest-classifier",
        ...     cv_slug="stratified-k-fold",
        ...     mode="with-evaluation"
        ... )
        >>> print(dirs['base'])
        /path/to/output/trainings/model-with-evaluation/random-forest-classifier/stratified-k-fold
    """
    mode_dir = f"model-{mode}"
    base_dir = os.path.join(root_dir, "trainings", mode_dir, classifier_slug, cv_slug)

    directories = {
        "base": base_dir,
        "features": os.path.join(base_dir, "features"),
        "significant_features": os.path.join(
            base_dir, "features", "significant_features"
        ),
        "models": os.path.join(base_dir, "models"),
    }

    if mode == "with-evaluation":
        directories["metrics"] = os.path.join(base_dir, "metrics")
        directories["figures"] = os.path.join(base_dir, "figures")

    # Create all directories
    for dir_path in directories.values():
        ensure_dir(dir_path)

    return directories
