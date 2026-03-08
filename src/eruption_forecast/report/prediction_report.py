"""Prediction report: probability time-series with uncertainty band and consensus."""

import json
from typing import Any

import pandas as pd

from eruption_forecast.config.constants import ERUPTION_PROBABILITY_THRESHOLD
from eruption_forecast.report.base_report import BaseReport


class PredictionReport(BaseReport):
    """Generate an interactive HTML report for eruption probability forecasts.

    Produces a Plotly probability time-series chart with:

    - Per-classifier dashed probability lines
    - Consensus solid line (black)
    - Shaded uncertainty band (mean ± std)
    - Vertical eruption event markers

    The input DataFrame is expected to follow the format produced by
    ``ModelPredictor.predict_proba()``.

    Args:
        prediction_df (pd.DataFrame | str): Prediction DataFrame (with a
            DatetimeIndex) or path to a prediction CSV file.
        eruption_dates (list[str] | None): Eruption dates in YYYY-MM-DD
            format, shown as vertical markers. Defaults to None.
        threshold (float): Decision threshold shown as a horizontal dashed
            line. Defaults to ``ERUPTION_PROBABILITY_THRESHOLD``.
        output_dir (str | None): Output directory for the saved report.
        root_dir (str | None): Anchor directory for path resolution.
    """

    def __init__(
        self,
        prediction_df: "pd.DataFrame | str",
        eruption_dates: list[str] | None = None,
        threshold: float = ERUPTION_PROBABILITY_THRESHOLD,
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize PredictionReport.

        Loads a prediction CSV when a path string is supplied; otherwise uses
        the provided DataFrame directly.

        Args:
            prediction_df (pd.DataFrame | str): Prediction DataFrame or CSV
                path.
            eruption_dates (list[str] | None): Eruption date strings
                (YYYY-MM-DD) for vertical marker annotations. Defaults to None.
            threshold (float): Decision threshold for the horizontal line.
                Defaults to ``ERUPTION_PROBABILITY_THRESHOLD``.
            output_dir (str | None): Directory for saved HTML reports.
                Defaults to None.
            root_dir (str | None): Anchor for path resolution. Defaults to None.
        """
        super().__init__(output_dir=output_dir, root_dir=root_dir)
        if isinstance(prediction_df, str):
            self._df = pd.read_csv(prediction_df, index_col=0, parse_dates=True)
            self._df.index = pd.to_datetime(self._df.index, errors="coerce")
        else:
            self._df = prediction_df
        self._eruption_dates = eruption_dates or []
        self._threshold = threshold

    @property
    def title(self) -> str:
        """Report title.

        Returns:
            str: Title string for the prediction report.
        """
        return "Eruption Probability Forecast Report"

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename.

        Returns:
            str: Template filename.
        """
        return "prediction.html.j2"

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _forecast_chart_json(self) -> str:
        """Build Plotly forecast chart JSON.

        Extracts per-classifier probability columns (named
        ``*_eruption_probability``) and the consensus columns from the
        prediction DataFrame. Renders them as a layered chart with:

        - Dashed traces per classifier
        - Solid black consensus line
        - Light-blue uncertainty band (consensus ± std)
        - Vertical eruption markers

        Returns:
            str: Plotly JSON string.
        """
        df = self._df
        if df.empty:
            return json.dumps({"data": [], "layout": {}})

        xs = [str(t)[:19] for t in df.index]
        traces: list[dict[str, Any]] = []
        shapes: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []

        # Per-classifier probability lines
        prob_cols = [c for c in df.columns if c.endswith("_eruption_probability")]
        std_col = next((c for c in df.columns if "consensus_uncertainty" in c), None)
        consensus_col = next(
            (c for c in df.columns if "consensus_eruption_probability" in c), None
        )

        for col in prob_cols:
            clf_label = col.replace("_eruption_probability", "").replace("_", " ")
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": clf_label,
                    "x": xs,
                    "y": df[col].tolist(),
                    "line": {"dash": "dash", "width": 1.2},
                    "opacity": 0.6,
                }
            )

        # Uncertainty band
        if consensus_col and std_col:
            mean_vals = df[consensus_col].tolist()
            std_vals = df[std_col].tolist()
            upper = [m + s for m, s in zip(mean_vals, std_vals, strict=False)]
            lower = [m - s for m, s in zip(mean_vals, std_vals, strict=False)]

            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Upper bound",
                    "x": xs,
                    "y": upper,
                    "line": {"width": 0},
                    "showlegend": False,
                }
            )
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Uncertainty band",
                    "x": xs,
                    "y": lower,
                    "fill": "tonexty",
                    "fillcolor": "rgba(59,130,246,0.12)",
                    "line": {"width": 0},
                }
            )

            # Consensus line
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Consensus",
                    "x": xs,
                    "y": mean_vals,
                    "line": {"color": "#0d1117", "width": 2.5},
                }
            )

        # Threshold line
        shapes.append(
            {
                "type": "line",
                "x0": xs[0] if xs else 0,
                "x1": xs[-1] if xs else 1,
                "y0": self._threshold,
                "y1": self._threshold,
                "xref": "x",
                "yref": "y",
                "line": {"color": "#dc2626", "width": 1.5, "dash": "longdash"},
            }
        )

        # Eruption markers
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
                    "line": {"color": "#b91c1c", "width": 2, "dash": "dot"},
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
                    "font": {"size": 9, "color": "#b91c1c"},
                    "yanchor": "bottom",
                }
            )

        layout = {
            "height": 400,
            "margin": {"l": 55, "r": 20, "t": 30, "b": 70},
            "xaxis": {"title": "DateTime", "rangeslider": {"visible": True}},
            "yaxis": {"title": "Eruption Probability", "range": [-0.05, 1.1]},
            "legend": {"orientation": "h", "y": -0.45},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
            "shapes": shapes,
            "annotations": annotations,
        }
        return json.dumps({"data": traces, "layout": layout})

    def _collect_data(self) -> dict[str, Any]:
        """Collect all template variables for the prediction report.

        Returns:
            dict[str, Any]: Context dict with date range, column metadata,
                eruption dates, threshold, and forecast chart JSON.
        """
        df = self._df

        def _fmt_date(val: Any) -> str:
            """Format an index value as a date string.

            Returns the date portion of a datetime-like value, or the raw
            string representation as a fallback for non-datetime indices.

            Args:
                val (Any): Index value to format.

            Returns:
                str: Formatted date string or ``"—"`` on failure.
            """
            try:
                return str(val.date())
            except AttributeError:
                return str(val)

        start = _fmt_date(df.index[0]) if not df.empty else "—"
        end = _fmt_date(df.index[-1]) if not df.empty else "—"

        clf_cols = [c for c in df.columns if c.endswith("_eruption_probability")]
        classifiers = [c.replace("_eruption_probability", "") for c in clf_cols]

        return {
            "start_date": start,
            "end_date": end,
            "classifiers": classifiers,
            "n_classifiers": len(classifiers),
            "eruption_dates": self._eruption_dates,
            "threshold": self._threshold,
            "forecast_chart_json": self._forecast_chart_json(),
        }
