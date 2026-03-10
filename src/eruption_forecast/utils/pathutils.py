"""File path resolution utilities.

This module provides utilities for resolving output directory paths relative
to a root anchor directory, creating directories, and loading JSON files.
"""

import os
import json
from typing import Any


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

