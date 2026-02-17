"""Nature/Science journal publication-quality styling for plots.

This module provides consistent styling configurations following Nature and Science
journal standards for scientific visualization. It includes color palettes, typography
settings, figure dimensions, and context managers for applying styles.

Color Palettes:
    - NATURE_COLORS: High-contrast primary colors for data visualization
    - OKABE_ITO: Colorblind-safe categorical palette (8 colors)
    - SEQUENTIAL: Perceptually uniform sequential colormap names
    - DIVERGING: Diverging colormap for heatmaps with neutral midpoint

Typography:
    - Sans-serif fonts (Arial/Helvetica family)
    - Font sizes: 8-12pt for various elements
    - High DPI outputs (150-300)

Layout:
    - Clean axis spines (remove top/right)
    - Light grid lines (gray, alpha=0.3)
    - Tight bounding boxes
    - Standard column widths for journals

Example:
    >>> import matplotlib.pyplot as plt
    >>> from eruption_forecast.plots.styles import apply_nature_style
    >>>
    >>> with apply_nature_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3], [1, 4, 9])
    ...     plt.savefig("output.png")
"""

from typing import Any, Literal
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


# Color Palettes
# Nature/Science standard colors - high contrast, colorblind-friendly
NATURE_COLORS = {
    "blue": "#1f77b4",  # Primary data color (negative class)
    "orange": "#ff7f0e",  # Secondary data color
    "red": "#d62728",  # Positive class / alert color
    "green": "#2ca02c",  # Success / validation
    "purple": "#9467bd",  # Alternative category
    "brown": "#8c564b",  # Neutral category
    "pink": "#e377c2",  # Highlight
    "gray": "#7f7f7f",  # Neutral / reference
    "olive": "#bcbd22",  # Alternative
    "cyan": "#17becf",  # Alternative
}

# Okabe-Ito colorblind-safe palette (widely used in scientific publications)
OKABE_ITO = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
    "#009E73",  # Bluish green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish purple
    "#000000",  # Black
]

# Perceptually uniform sequential colormaps
SEQUENTIAL = ["viridis", "plasma", "inferno", "cividis"]

# Diverging colormaps for heatmaps (zero-centered)
DIVERGING = ["RdYlBu_r", "RdBu_r", "coolwarm", "seismic"]

# Typography Configuration
FONT_FAMILY = "sans-serif"
FONT_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]

FONT_SIZES = {
    "title": 12,  # Plot title
    "label": 10,  # Axis labels
    "tick": 8,  # Tick labels
    "legend": 8,  # Legend text
    "annotation": 9,  # Annotations
}

# Figure Dimensions (inches) - standard journal column widths
FIGURE_SIZES = {
    "single_column": (3.5, 2.625),  # Nature single column: 89mm (3.5")
    "double_column": (7.0, 5.25),  # Nature double column: 178mm (7")
    "full_page": (7.0, 9.0),  # Full page height
    "square": (5.0, 5.0),  # Square format
    "wide": (10.0, 3.0),  # Wide time-series format
    "tall": (4.0, 8.0),  # Tall format for vertical bar charts
}

# DPI Settings
DPI_SCREEN = 100  # For screen display
DPI_PRINT = 150  # Standard print quality
DPI_PUBLICATION = 300  # High-quality publication

# Grid and Spine Configuration
GRID_CONFIG = {
    "linewidth": 0.5,
    "linestyle": "--",
    "alpha": 0.3,
    "color": "gray",
}

SPINE_CONFIG: dict[str, Any] = {
    "linewidth": 1.0,
    "color": "black",
}


def setup_nature_style() -> dict:
    """Create matplotlib rcParams dict for Nature/Science journal style.

    Generates a complete configuration dictionary with publication-quality settings
    including fonts, figure dimensions, axes styling, and color palettes.

    Returns:
        dict: Matplotlib rcParams configuration dictionary containing keys for
            font settings, figure DPI, axes styling, grid configuration, and
            color cycles.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> style = setup_nature_style()
        >>> plt.rcParams.update(style)
        >>> # Now all plots will use Nature/Science styling
    """
    return {
        # Font configuration
        "font.family": FONT_FAMILY,
        "font.sans-serif": FONT_SANS_SERIF,
        "font.size": FONT_SIZES["label"],
        "axes.labelsize": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        # Figure configuration
        "figure.dpi": DPI_SCREEN,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        # Axes configuration
        "axes.linewidth": SPINE_CONFIG["linewidth"],
        "axes.edgecolor": SPINE_CONFIG["color"],
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid configuration
        "grid.linewidth": GRID_CONFIG["linewidth"],
        "grid.linestyle": GRID_CONFIG["linestyle"],
        "grid.alpha": GRID_CONFIG["alpha"],
        "grid.color": GRID_CONFIG["color"],
        # Line configuration
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        # Tick configuration
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend configuration
        "legend.frameon": False,
        "legend.loc": "best",
        # Saving configuration
        "savefig.dpi": DPI_PRINT,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }


@contextmanager
def apply_nature_style():
    """Context manager to temporarily apply Nature/Science journal style.

    Applies publication-quality styling within the context and restores
    previous matplotlib settings upon exit. Safe to nest and won't leak settings.

    Yields:
        None: Context manager yields nothing but manages rcParams state.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> with apply_nature_style():
        ...     fig, ax = plt.subplots()
        ...     ax.plot([1, 2, 3], [1, 4, 9])
        ...     plt.savefig("styled_plot.png")
        >>> # Previous matplotlib settings are now restored
    """
    original_params = mpl.rcParams.copy()
    try:
        plt.rcParams.update(setup_nature_style())
        yield
    finally:
        mpl.rcParams.update(original_params)


def get_color(
    name: str,
    palette: Literal["nature", "okabe_ito"] = "nature",
) -> str:
    """Get a color from a named palette.

    Retrieves a hex color code from either the NATURE_COLORS or OKABE_ITO palette.
    For Nature palette, use color names. For Okabe-Ito, use numeric indices (0-7).

    Args:
        name (str): Color name (for "nature") like "blue", "red", "green", etc.,
            or numeric index as string (for "okabe_ito") like "0", "1", ..., "7".
        palette (Literal["nature", "okabe_ito"], optional): Palette identifier.
            Use "nature" for named colors or "okabe_ito" for colorblind-safe
            indexed palette. Defaults to "nature".

    Returns:
        str: Hex color code (e.g., "#1f77b4").

    Raises:
        ValueError: If color name is not in NATURE_COLORS, if index is out of
            range for OKABE_ITO (must be 0-7), or if palette name is invalid.

    Examples:
        >>> get_color("blue", "nature")
        '#1f77b4'
        >>> get_color("0", "okabe_ito")  # First Okabe-Ito color (orange)
        '#E69F00'
        >>> get_color("4", "okabe_ito")  # Blue
        '#0072B2'
    """
    if palette == "nature":
        if name not in NATURE_COLORS:
            msg = f"Color '{name}' not in NATURE_COLORS. Available: {list(NATURE_COLORS.keys())}"
            raise ValueError(msg)
        return NATURE_COLORS[name]
    elif palette == "okabe_ito":
        try:
            index = int(name)
            return OKABE_ITO[index]
        except (ValueError, IndexError) as e:
            msg = f"Invalid Okabe-Ito index '{name}'. Must be 0-{len(OKABE_ITO) - 1}"
            raise ValueError(msg) from e
    else:
        msg = f"Unknown palette '{palette}'. Use 'nature' or 'okabe_ito'"
        raise ValueError(msg)


def configure_spine(ax: plt.Axes, which: list[str] | None = None) -> None:
    """Configure axis spines for publication quality.

    Removes top and right spines by default (Nature/Science standard). Visible
    spines are styled with consistent linewidth and color.

    Args:
        ax (plt.Axes): Matplotlib axes object to configure.
        which (list[str] | None, optional): List of spines to keep. Valid values
            are "left", "right", "top", "bottom". If None, keeps only "left" and
            "bottom" (standard Nature/Science format). Defaults to None.

    Returns:
        None: Modifies axes in-place.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> configure_spine(ax)  # Removes top and right spines
        >>> configure_spine(ax, which=["left", "bottom", "right"])  # Keep right spine
    """
    if which is None:
        which = ["left", "bottom"]

    all_spines = ["top", "bottom", "left", "right"]
    for spine in all_spines:
        if spine in which:
            ax.spines[spine].set_linewidth(SPINE_CONFIG["linewidth"])
            ax.spines[spine].set_color(SPINE_CONFIG["color"])
        else:
            ax.spines[spine].set_visible(False)


def get_figure_size(
    size_key: Literal[
        "single_column", "double_column", "full_page", "square", "wide", "tall"
    ],
) -> tuple[float, float]:
    """Get predefined figure size for journal publications.

    Returns standard figure dimensions matching Nature/Science journal column widths.
    Single column = 89mm (3.5"), double column = 178mm (7").

    Args:
        size_key (Literal["single_column", "double_column", "full_page", "square", "wide", "tall"]):
            Key for predefined figure size. Options are:
            - "single_column": 3.5" × 2.625" (Nature single column width)
            - "double_column": 7.0" × 5.25" (Nature double column width)
            - "full_page": 7.0" × 9.0" (full page height)
            - "square": 5.0" × 5.0" (square format)
            - "wide": 10.0" × 3.0" (wide time-series format)
            - "tall": 4.0" × 8.0" (tall vertical bar chart format)

    Returns:
        tuple[float, float]: Figure size as (width, height) in inches.

    Examples:
        >>> get_figure_size("single_column")
        (3.5, 2.625)
        >>> get_figure_size("double_column")
        (7.0, 5.25)
        >>> get_figure_size("wide")
        (10.0, 3.0)
    """
    return FIGURE_SIZES[size_key]
