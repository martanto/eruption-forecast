"""Label data report: window config, class distribution, and eruption timeline."""

import json
from typing import Any

import pandas as pd

from eruption_forecast.label.label_data import LabelData
from eruption_forecast.report.base_report import BaseReport


class LabelReport(BaseReport):
    """Generate an interactive HTML report for label data.

    Produces a self-contained HTML page with:

    - Window configuration parameters parsed from the label filename
    - Class distribution Plotly bar chart (eruption vs non-eruption)
    - Counts table (total / eruption / non-eruption windows, imbalance ratio)
    - Label timeline chart with eruption date markers

    Args:
        label_data (LabelData | str): A :class:`LabelData` instance, or a
            path to a label CSV file which will be loaded automatically.
        eruption_dates (list[str] | None): Eruption dates in YYYY-MM-DD format
            shown as vertical markers on the timeline chart.
        output_dir (str | None): Output directory for the saved report.
        root_dir (str | None): Anchor directory for path resolution.
    """

    def __init__(
        self,
        label_data: "LabelData | str",
        eruption_dates: list[str] | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize LabelReport with label data.

        Loads a label CSV when a path string is supplied; otherwise wraps the
        provided :class:`LabelData` instance.

        Args:
            label_data (LabelData | str): LabelData instance or path to a
                label CSV file.
            eruption_dates (list[str] | None): Optional list of eruption
                dates (YYYY-MM-DD) for timeline markers. Defaults to None.
            output_dir (str | None): Directory for saved HTML reports.
                Defaults to None.
            root_dir (str | None): Anchor directory for relative path
                resolution. Defaults to None.
        """
        super().__init__(output_dir=output_dir, root_dir=root_dir)
        if isinstance(label_data, str):
            self._label = LabelData(label_data)
        else:
            self._label = label_data
        self._eruption_dates = eruption_dates or []

    @property
    def title(self) -> str:
        """Report title.

        Returns:
            str: Title string for the label report.
        """
        return "Label Data Report"

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename.

        Returns:
            str: Template filename.
        """
        return "label.html.j2"

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _distribution_chart_json(self, df: pd.DataFrame) -> str:
        """Build Plotly bar chart JSON for eruption vs non-eruption class counts.

        Args:
            df (pd.DataFrame): Label DataFrame with an 'is_erupted' column.

        Returns:
            str: Plotly JSON string for a bar chart.
        """
        counts = df["is_erupted"].value_counts().sort_index()
        labels = ["Non-eruption (0)", "Eruption (1)"]
        values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
        colors = ["#64748b", "#dc2626"]

        traces = [
            {
                "type": "bar",
                "x": labels,
                "y": values,
                "marker": {"color": colors},
                "text": [str(v) for v in values],
                "textposition": "outside",
            }
        ]
        layout = {
            "height": 280,
            "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
            "yaxis": {"title": "Window count"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
            "showlegend": False,
        }
        return json.dumps({"data": traces, "layout": layout})

    def _timeline_chart_json(self, df: pd.DataFrame) -> str:
        """Build Plotly timeline chart with eruption label markers.

        Shows ``is_erupted`` value over time and overlays vertical lines for
        each eruption date provided at construction.

        Args:
            df (pd.DataFrame): Label DataFrame with DatetimeIndex and
                'is_erupted' column.

        Returns:
            str: Plotly JSON string with scatter trace and shape annotations.
        """
        traces = [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "is_erupted",
                "x": [str(t)[:19] for t in df.index],
                "y": df["is_erupted"].tolist(),
                "line": {"color": "#1a3a5c", "width": 1},
                "fill": "tozeroy",
                "fillcolor": "rgba(26,58,92,0.08)",
            }
        ]

        shapes = []
        annotations = []
        for date in self._eruption_dates:
            shapes.append(
                {
                    "type": "line",
                    "x0": date,
                    "x1": date,
                    "y0": 0,
                    "y1": 1,
                    "xref": "x",
                    "yref": "paper",
                    "line": {"color": "#dc2626", "width": 2, "dash": "dash"},
                }
            )
            annotations.append(
                {
                    "x": date,
                    "y": 1,
                    "xref": "x",
                    "yref": "paper",
                    "text": f"Eruption<br>{date}",
                    "showarrow": False,
                    "font": {"size": 9, "color": "#dc2626"},
                    "yanchor": "bottom",
                }
            )

        layout = {
            "height": 260,
            "margin": {"l": 50, "r": 20, "t": 20, "b": 60},
            "xaxis": {"title": "DateTime"},
            "yaxis": {"title": "is_erupted", "range": [-0.05, 1.3]},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
            "shapes": shapes,
            "annotations": annotations,
        }
        return json.dumps({"data": traces, "layout": layout})

    def _collect_data(self) -> dict[str, Any]:
        """Collect all template variables for the label report.

        Returns:
            dict[str, Any]: Context dict containing window parameters,
                class counts, and Plotly chart JSON strings.
        """
        df = self._label.df
        total = len(df)
        n_eruption = int(df["is_erupted"].sum())
        n_non_eruption = total - n_eruption
        ratio = (
            round(n_non_eruption / n_eruption, 2) if n_eruption > 0 else float("inf")
        )

        return {
            "params": self._label.parameters,
            "total_windows": total,
            "n_eruption": n_eruption,
            "n_non_eruption": n_non_eruption,
            "imbalance_ratio": ratio,
            "eruption_dates": self._eruption_dates,
            "distribution_chart_json": self._distribution_chart_json(df),
            "timeline_chart_json": self._timeline_chart_json(df),
        }
