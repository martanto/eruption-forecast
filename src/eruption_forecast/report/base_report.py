"""Abstract base class for all pipeline report generators.

Each concrete report subclass collects data, renders a Jinja2 template, and
saves a self-contained HTML file (with optional PDF export via weasyprint).
"""

import os
from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape

from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir


_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("eruption-forecast")
except Exception:
    _VERSION = "unknown"


class BaseReport(ABC):
    """Abstract base for all eruption-forecast HTML report generators.

    Subclasses implement :meth:`title` and :meth:`_collect_data` to provide
    page-specific content; the shared :meth:`to_html` / :meth:`save` /
    :meth:`to_pdf` machinery handles rendering and persistence.

    Args:
        output_dir (str | None): Destination directory for saved reports.
            Resolved against ``root_dir`` using :func:`resolve_output_dir`.
        root_dir (str | None): Anchor directory for path resolution.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize the report with resolved output directory.

        Resolves ``output_dir`` relative to ``root_dir`` (or ``os.getcwd()``
        when both are None) and configures the Jinja2 template environment.

        Args:
            output_dir (str | None): Output directory for saved report files.
                Defaults to None (resolved from root_dir).
            root_dir (str | None): Anchor directory for relative path
                resolution. Defaults to None (uses os.getcwd()).
        """
        self.output_dir = resolve_output_dir(
            output_dir, root_dir, os.path.join("output", "reports")
        )
        self._env = Environment(
            loader=FileSystemLoader(_TEMPLATES_DIR),
            autoescape=select_autoescape(["html"]),
        )

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable title for this report.

        Returns:
            str: Report title displayed in the HTML header.
        """
        ...

    @abstractmethod
    def _collect_data(self) -> dict[str, Any]:
        """Collect all template variables for rendering.

        Subclasses gather stats, build Plotly JSON, and return a dict that
        is merged with the shared base variables before template rendering.

        Returns:
            dict[str, Any]: Template context variables specific to this report.
        """
        ...

    # ------------------------------------------------------------------
    # Shared template variables
    # ------------------------------------------------------------------

    def _base_context(self) -> dict[str, Any]:
        """Build shared template variables injected into every report.

        Returns:
            dict[str, Any]: Dict containing title, generated_at, version,
                and an empty sections list.
        """
        return {
            "title": self.title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": _VERSION,
            "sections": [],
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename used by this report.

        Returns:
            str: Template filename relative to the templates directory.
        """
        raise NotImplementedError(
            "Subclasses must define _template_name or override to_html()."
        )

    def to_html(self) -> str:
        """Render the report to an HTML string.

        Merges the base context with subclass-specific data and renders
        the Jinja2 template.

        Returns:
            str: Complete self-contained HTML document as a string.
        """
        ctx = {**self._base_context(), **self._collect_data()}
        template = self._env.get_template(self._template_name)
        return template.render(**ctx)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str | None = None) -> str:
        """Render and save the report to an HTML file.

        Creates ``output_dir`` if it does not exist, then writes the rendered
        HTML. The default filename is derived from the class name.

        Args:
            filename (str | None): Output filename (basename only, not a full
                path). If None, uses a snake-case version of the class name.

        Returns:
            str: Absolute path to the written HTML file.
        """
        ensure_dir(self.output_dir)
        if filename is None:
            cls_name = type(self).__name__.lower().replace("report", "_report")
            filename = f"{cls_name}.html"
        if not filename.endswith(".html"):
            filename = f"{filename}.html"
        path = os.path.join(self.output_dir, filename)
        html = self.to_html()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path

    def to_pdf(self, path: str) -> str:
        """Export the report to PDF using weasyprint.

        Renders the HTML report and converts it to PDF. Requires the optional
        ``weasyprint`` package; raises ``ImportError`` with a clear message
        when it is not installed.

        Args:
            path (str): Destination path for the PDF file (including extension).

        Returns:
            str: The ``path`` argument, for convenience.

        Raises:
            ImportError: If ``weasyprint`` is not installed.
        """
        try:
            from weasyprint import HTML  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PDF export requires weasyprint. Install it with: uv add weasyprint"
            ) from exc

        html_str = self.to_html()
        ensure_dir(os.path.dirname(path) or ".")
        HTML(string=html_str).write_pdf(path)
        return path
