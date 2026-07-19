import os
import json
import pickle
from typing import Any
from importlib.metadata import metadata

import joblib
import matplotlib
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger


def pdf_metadata(title: str | None = None) -> dict[str, str]:
    """Build PDF metadata dict from package metadata and environment.

    Reads package version and homepage URL from ``importlib.metadata``.
    The ``Author`` field is resolved from the environment in priority order:
    ``GIT_AUTHOR_NAME`` → ``USERNAME`` (Windows) → ``USER`` (Unix) →
    ``"eruption-forecast"`` as a last-resort fallback.

    Returns:
        dict[str, str]: Metadata dict suitable for passing to
        ``matplotlib``'s ``savefig(metadata=...)``.  Keys: ``Title``,
        ``Author``, ``Subject``, ``Keywords``, ``Creator``.
    """
    package_metadata = metadata("eruption-forecast")
    version: str = package_metadata["Version"]

    _pdf_metadata = {
        "Title": title or "Eruption probability forecast",
        "Author": (
            os.environ.get("GIT_AUTHOR_NAME")
            or os.environ.get("USERNAME")
            or os.environ.get("USER")
            or "eruption-forecast"
        ),
        "Subject": "Eruption probability forecast",
        "Keywords": "eruption, forecast, seismic, tremor",
        "Creator": f"eruption-forecast v{version}",
    }

    return _pdf_metadata


def ensure_dir(path: str) -> str:
    """Create a directory (and any missing parents) if it does not already exist.

    A thin, named wrapper around ``os.makedirs(path, exist_ok=True)`` that
    returns the path so callers can chain it inline.

    Args:
        path (str): Directory path to create.

    Returns:
        str: The same ``path`` that was passed in.
    """
    if path:
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


def load_pickle(path: str) -> Any:
    """Load a joblib pickle and surface the file path on failure.

    Thin wrapper around ``joblib.load`` that fails loudly with the
    offending file path. If the file is missing, raises
    ``FileNotFoundError`` naming ``path``. If ``joblib.load`` raises an
    unpickle-related error (commonly triggered by a package upgrade that
    moves or removes a class), the original exception is chained via
    ``raise ... from e`` so the traceback still shows *why* it failed
    while the new top-level ``RuntimeError`` shows *which* file failed.

    Args:
        path (str): Path to a ``.pkl`` file produced by ``joblib.dump``.

    Returns:
        Any: The deserialised object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If ``joblib.load`` raises an unpickle-related error.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    try:
        return joblib.load(path)
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        EOFError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as e:
        raise RuntimeError(f"Failed to load pickle: {path}") from e


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


def save_figure_as_pdf(
    fig: plt.Figure,
    filepath: str,
    title: str | None = None,
    *,
    verbose: bool = True,
) -> None:
    """Save a matplotlib figure as a PDF with embedded TrueType fonts.

    Wraps ``fig.savefig`` inside a scoped ``matplotlib.rc_context`` that
    sets ``pdf.fonttype=42`` so text in the PDF stays selectable and
    renders consistently across viewers and vector editors. Embeds
    package metadata via :func:`pdf_metadata` and creates the parent
    directory if it does not already exist. The figure is NOT closed —
    callers that render PNG and PDF from the same figure should close
    it themselves.

    Args:
        fig (plt.Figure): Figure to save.
        filepath (str): Destination path, including the ``.pdf`` extension.
        title (str | None, optional): Title embedded in the PDF metadata.
            Falls back to the :func:`pdf_metadata` default when ``None``.
            Defaults to ``None``.
        verbose (bool, optional): When ``True``, log an info message
            after saving. Defaults to ``True``.

    Returns:
        None
    """
    ensure_dir(os.path.dirname(filepath))

    # Type 42 embeds TrueType fonts — text stays selectable and
    # renders consistently in all PDF viewers and vector editors.
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(
            filepath,
            bbox_inches="tight",
            facecolor="white",
            edgecolor=None,
            metadata=pdf_metadata(title=title),
        )

    if verbose:
        logger.info(f"Saved: {filepath}")


def save_figure(
    fig: plt.Figure,
    filepath: str,
    dpi: int,
    *,
    filetype: str = "png",
    save_as_pdf: bool = False,
    pdf_title: str | None = None,
    verbose: bool = True,
) -> None:
    """Save a matplotlib figure to disk and always close it.

    Accepts ``filepath`` either with or without a trailing extension. When
    the filename already ends in ``.{filetype}`` it is used as-is; otherwise
    the extension is appended. A directory in the path that contains a dot
    (e.g. ``output/foo.png-debug/figure``) is not mistaken for an extension.
    When ``save_as_pdf`` is ``True``, a PDF sibling with embedded TrueType
    fonts and PDF metadata is also written via :func:`save_figure_as_pdf`
    before the figure is closed; the sibling shares the primary file's stem
    (e.g. ``output/foo.png`` → ``output/foo.pdf``). The figure is closed
    unconditionally after both saves so it cannot be reused by subsequent
    plot calls.

    Args:
        fig (plt.Figure): Figure to save and close.
        filepath (str): Destination path, with or without the trailing
            ``.{filetype}`` extension.
        dpi (int): Resolution in dots per inch.
        filetype (str): File extension without a leading dot. Defaults to
            ``"png"``.
        save_as_pdf (bool, optional): When ``True``, also write a PDF
            sibling with embedded TrueType fonts and PDF metadata via
            :func:`save_figure_as_pdf`. The sibling reuses the primary
            file's stem with the ``.pdf`` extension. Intended for adding a
            vector-editor-friendly companion next to a raster primary
            output; pass ``filetype="pdf"`` instead when only a PDF is
            needed. Defaults to ``False``.
        pdf_title (str | None, optional): Title embedded in the PDF
            metadata when ``save_as_pdf=True``. Ignored otherwise. Defaults
            to ``None``.
        verbose (bool): When True, log an info message after saving.
            Defaults to True.

    Returns:
        None
    """
    existing_ext = os.path.splitext(filepath)[1].lstrip(".").lower()
    full_path = (
        filepath if existing_ext == filetype.lower() else f"{filepath}.{filetype}"
    )

    ensure_dir(os.path.dirname(full_path))

    fig.savefig(
        full_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor=None
    )

    if verbose:
        logger.info(f"Saved: {full_path}")

    if save_as_pdf:
        pdf_path = f"{os.path.splitext(full_path)[0]}.pdf"
        save_figure_as_pdf(
            fig=fig,
            filepath=pdf_path,
            title=pdf_title,
            verbose=verbose,
        )

    plt.close(fig)


def save_data(data: Any, path: str, *, filetype: str = "csv") -> None:
    """Persist a data object to disk.

    Dispatches on ``filetype``:

    - ``"csv"`` — writes a ``pd.DataFrame`` via ``to_csv``.
    - ``"parquet"`` — writes a ``pd.DataFrame`` via ``to_parquet`` using
      the ``pyarrow`` engine with Snappy compression and ``index=False``.
      Preserves column dtypes and typically compresses long-form tables
      several times smaller than CSV.
    - anything else — serialises the object with ``joblib.dump``.

    The destination directory is created automatically if absent.

    Args:
        data (Any): Object to persist. Pass a ``pd.DataFrame`` with
            ``filetype="csv"`` or ``filetype="parquet"``; pass any
            joblib-serialisable object with ``filetype="pkl"``.
        path (str): Destination path WITHOUT a file extension.
        filetype (str): File extension without a leading dot, also
            determines the serialisation method. Defaults to ``"csv"``.

    Returns:
        None
    """
    full_path = f"{path}.{filetype}"
    ensure_dir(os.path.dirname(full_path))
    if filetype == "csv":
        data.to_csv(full_path)
    elif filetype == "parquet":
        data.to_parquet(
            full_path, engine="pyarrow", compression="snappy", index=False
        )
    else:
        joblib.dump(data, full_path)
    logger.info(f"Saved: {full_path}")


def setup_nslc_directories(
    network: str,
    station: str,
    location: str,
    channel: str,
    output_dir: str | None = None,
    root_dir: str | None = None,
) -> tuple[str, str, str]:
    """Set up directory structure for forecast model outputs.

    Creates the NSLC (Network.Station.Location.Channel) identifier and
    builds the directory structure for storing model outputs.

    Args:
        network (str): Network code (e.g., "VG").
        station (str): Station code (e.g., "OJN").
        location (str): Location code (e.g., "00").
        channel (str): Channel code (e.g., "EHZ").
        output_dir (str | None): Base output directory. If None, defaults to
            "output" relative to root_dir.
        root_dir (str | None, optional): Anchor directory for resolving relative
            paths. If None, falls back to ``os.getcwd()``. Defaults to None.

    Returns:
        tuple[str, str, str]: A 3-tuple containing:

            - **nslc** (str): Combined identifier (e.g., "VG.OJN.00.EHZ")
            - **output_dir** (str): Resolved output directory path
            - **station_dir** (str): Station-specific directory path
    """
    nslc = f"{network}.{station}.{location}.{channel}"
    output_dir = resolve_output_dir(output_dir, root_dir, "output")
    station_dir = os.path.join(output_dir, nslc.upper())

    return nslc, output_dir, station_dir


def generate_features_filepaths(
    random_state: int,
    features_seed_dir: str,
    features_resampled_dir: str,
    figures_seed_dir: str,
    plot_features: bool = False,
    overwrite: bool = False,
) -> tuple[bool, str, str, str | None]:
    filename = f"{random_state:05d}"

    # Required
    features_seed_path = os.path.join(features_seed_dir, f"{filename}.csv")
    features_resampled_path = os.path.join(features_resampled_dir, f"{filename}.csv")

    filepath_required = [features_seed_path, features_resampled_path]

    can_skip = not overwrite and all(os.path.isfile(p) for p in filepath_required)

    # Optional
    figures_seed_path = (
        os.path.join(figures_seed_dir, f"{filename}.png") if plot_features else None
    )
    if can_skip and plot_features:
        can_skip = figures_seed_path is not None and os.path.isfile(figures_seed_path)

    return (
        can_skip,
        features_seed_path,
        features_resampled_path,
        figures_seed_path,
    )
