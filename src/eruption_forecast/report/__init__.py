"""Report generation package for the eruption-forecast pipeline.

Generates self-contained interactive HTML reports (backed by Plotly and CDN JS)
for each stage of the forecasting pipeline, as well as a combined full-pipeline
report with sidebar navigation. PDF export is optionally available via
``weasyprint``.

Public classes (one per pipeline stage):

- ``BaseReport`` — abstract base class; all section reports inherit from it.
- ``TremorReport`` — tremor completeness, gap statistics, and interactive charts.
- ``LabelReport`` — window configuration, class distribution, and eruption timeline.
- ``FeaturesReport`` — feature counts, selection method, and band contributions.
- ``TrainingReport`` — per-seed metrics, ROC, confusion matrix, and seed stability.
- ``ComparatorReport`` — side-by-side classifier comparison charts and tables.
- ``PredictionReport`` — probability time-series with uncertainty band and consensus.
- ``PipelineReport`` — combines all section reports; ``from_output_dir()`` scans an
  existing output directory and auto-detects which sections have artifacts available.

Standalone entry point:

- ``generate_report(output_dir, sections, fmt)`` — scans an output directory and
  builds a ``PipelineReport`` from available artifacts. Returns the path to the
  saved HTML (or PDF) file. Equivalent to ``ForecastModel.generate_report()``.
"""

from eruption_forecast.report.base_report import BaseReport
from eruption_forecast.report.label_report import LabelReport
from eruption_forecast.report.tremor_report import TremorReport
from eruption_forecast.report.features_report import FeaturesReport
from eruption_forecast.report.pipeline_report import PipelineReport
from eruption_forecast.report.training_report import TrainingReport
from eruption_forecast.report.comparator_report import ComparatorReport
from eruption_forecast.report.prediction_report import PredictionReport


def generate_report(
    output_dir: str,
    sections: list[str] | None = None,
    fmt: str = "html",
) -> str:
    """Generate a full pipeline report from an existing output directory.

    Scans ``output_dir`` for known artifact files (tremor CSV, label CSV,
    features CSV, metrics directory, prediction CSV) and builds a
    :class:`PipelineReport` from whichever sections have data available.

    Args:
        output_dir (str): Root output directory to scan (e.g. the
            station-specific directory ``output/VG.OJN.00.EHZ/``).
        sections (list[str] | None): Section names to include. If None, all
            available sections are included. Valid values: ``"tremor"``,
            ``"label"``, ``"features"``, ``"training"``, ``"prediction"``.
            Defaults to None.
        fmt (str): Output format — currently only ``"html"`` is supported.
            Pass ``"pdf"`` to export PDF (requires ``weasyprint``).
            Defaults to ``"html"``.

    Returns:
        str: Absolute path to the saved report file.

    Raises:
        ValueError: If an unsupported ``fmt`` is supplied.

    Examples:
        >>> path = generate_report("output/VG.OJN.00.EHZ")
        >>> path = generate_report("output/VG.OJN.00.EHZ", sections=["tremor", "label"])
    """
    pipeline = PipelineReport.from_output_dir(output_dir, sections=sections)

    if fmt == "html":
        return pipeline.save("pipeline_report.html")
    if fmt == "pdf":
        html_path = pipeline.save("pipeline_report.html")
        pdf_path = html_path.replace(".html", ".pdf")
        return pipeline.to_pdf(pdf_path)

    raise ValueError(f"Unsupported format '{fmt}'. Choose 'html' or 'pdf'.")


__all__ = [
    "BaseReport",
    "TremorReport",
    "LabelReport",
    "FeaturesReport",
    "TrainingReport",
    "ComparatorReport",
    "PredictionReport",
    "PipelineReport",
    "generate_report",
]
