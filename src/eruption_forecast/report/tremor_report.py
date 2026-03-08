"""Tremor data report: completeness, statistics, and interactive Plotly charts."""

import json
from typing import Any

import pandas as pd

from eruption_forecast.report.base_report import BaseReport
from eruption_forecast.tremor.tremor_data import TremorData


class TremorReport(BaseReport):
    """Generate an interactive HTML report for tremor time-series data.

    Produces a self-contained HTML page with:

    - Data completeness summary (% gaps, gap detection)
    - Per-band statistics table (min, max, mean, std)
    - Full-range overview Plotly chart (all frequency bands)
    - Per-day detail Plotly chart with a date dropdown selector

    Args:
        tremor_data (TremorData | str): A :class:`TremorData` instance, or a
            path to a tremor CSV file which will be loaded automatically.
        station (str | None): Station code for the report header. Defaults to None.
        output_dir (str | None): Output directory for the saved report.
        root_dir (str | None): Anchor directory for path resolution.
    """

    def __init__(
        self,
        tremor_data: "TremorData | str",
        station: str | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize TremorReport with tremor data.

        Loads a tremor CSV when a path string is supplied; otherwise wraps the
        provided :class:`TremorData` instance. Calls the parent constructor to
        resolve the output directory.

        Args:
            tremor_data (TremorData | str): TremorData instance or path to a
                tremor CSV file.
            station (str | None): Station code shown in the report header.
                Defaults to None.
            output_dir (str | None): Directory for saved HTML reports.
                Defaults to None.
            root_dir (str | None): Anchor directory for relative path
                resolution. Defaults to None.
        """
        super().__init__(output_dir=output_dir, root_dir=root_dir)
        if isinstance(tremor_data, str):
            td = TremorData()
            td.from_csv(tremor_data)
            self._tremor = td
        else:
            self._tremor = tremor_data
        self._station = station or ""

    @property
    def title(self) -> str:
        """Report title including station name.

        Returns:
            str: Title string for the tremor report.
        """
        suffix = f" — {self._station}" if self._station else ""
        return f"Tremor Data Report{suffix}"

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename.

        Returns:
            str: Template filename.
        """
        return "tremor.html.j2"

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------

    def _completeness(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute data completeness metrics for the tremor DataFrame.

        Calculates expected 10-minute intervals, compares against actual rows,
        and detects gaps where consecutive timestamps differ by more than the
        expected interval.

        Args:
            df (pd.DataFrame): Tremor DataFrame with DatetimeIndex.

        Returns:
            dict[str, Any]: Dict with keys: total_rows, expected_rows,
                missing_rows, completeness_pct, n_gaps, gap_list.
        """
        if df.empty:
            return {
                "total_rows": 0,
                "expected_rows": 0,
                "missing_rows": 0,
                "completeness_pct": 0.0,
                "n_gaps": 0,
                "gap_list": [],
            }

        total_minutes = int((df.index[-1] - df.index[0]).total_seconds() / 60)
        expected = max(1, total_minutes // 10 + 1)
        actual = len(df)
        missing = max(0, expected - actual)
        pct = round(actual / expected * 100, 2)

        # Detect gaps larger than 10 minutes
        diffs = df.index.to_series().diff().dropna()
        gap_threshold = pd.Timedelta(minutes=11)
        gaps = diffs[diffs > gap_threshold]
        gap_list = [
            {
                "start": str(ts - diff)[:19],
                "end": str(ts)[:19],
                "duration_min": round(diff.total_seconds() / 60, 1),
            }
            for ts, diff in gaps.items()
        ][:20]  # limit to 20 for the report

        return {
            "total_rows": actual,
            "expected_rows": expected,
            "missing_rows": missing,
            "completeness_pct": pct,
            "n_gaps": len(gap_list),
            "gap_list": gap_list,
        }

    def _band_stats(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Compute per-column statistics for tremor bands.

        Args:
            df (pd.DataFrame): Tremor DataFrame.

        Returns:
            list[dict[str, Any]]: One dict per column with keys: column, min,
                max, mean, std, null_count.
        """
        rows = []
        for col in df.columns:
            s = df[col].dropna()
            rows.append(
                {
                    "column": col,
                    "min": f"{s.min():.4g}" if len(s) else "—",
                    "max": f"{s.max():.4g}" if len(s) else "—",
                    "mean": f"{s.mean():.4g}" if len(s) else "—",
                    "std": f"{s.std():.4g}" if len(s) else "—",
                    "null_count": int(df[col].isna().sum()),
                }
            )
        return rows

    def _overview_charts_by_band(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Build one Plotly chart per frequency band column for the full-range overview.

        Each column (e.g. ``rsam_f0``, ``dsar_f0-f1``) gets its own chart so
        that amplitude scales are not mixed. Every chart includes a range slider
        for zooming.

        Args:
            df (pd.DataFrame): Tremor DataFrame with DatetimeIndex.

        Returns:
            list[dict[str, Any]]: List of dicts, each with keys ``label``
                (display title), ``id`` (HTML element id), and ``json``
                (Plotly JSON string).
        """
        if df.empty:
            return []

        xs = [str(t)[:19] for t in df.index]
        charts = []
        for col in df.columns:
            trace = {
                "type": "scatter",
                "mode": "lines",
                "name": col,
                "x": xs,
                "y": df[col].where(df[col].notna(), None).tolist(),
                "line": {"width": 1, "color": "#1a3a5c"},
            }
            layout = {
                "height": 260,
                "margin": {"l": 60, "r": 20, "t": 10, "b": 50},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
                "yaxis": {"title": col},
                "showlegend": False,
                "paper_bgcolor": "white",
                "plot_bgcolor": "#f7f9fc",
            }
            charts.append(
                {
                    "label": col,
                    "id": f"overview-{col.replace('/', '-')}",
                    "json": json.dumps({"data": [trace], "layout": layout}),
                }
            )
        return charts

    def _daily_charts_by_band(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Build per-band daily detail data for the date-selector charts.

        Returns one entry per column. Each entry contains the sorted list of
        available dates and a mapping of ``date → {x, y}`` so that the
        template can render a plain HTML ``<select>`` and update each Plotly
        chart via JavaScript without relying on Plotly's ``updatemenus``.

        Args:
            df (pd.DataFrame): Tremor DataFrame with DatetimeIndex.

        Returns:
            list[dict[str, Any]]: List of dicts with keys ``label``, ``id``,
                ``dates``, ``series_json`` (JSON mapping date → {x, y}), and
                ``layout_json`` (base Plotly layout JSON).
        """
        if df.empty:
            return []

        dates = sorted({str(t.date()) for t in df.index})
        charts = []
        for col in df.columns:
            series: dict[str, dict[str, list]] = {}
            for date in dates:
                day_df = df[df.index.date == pd.Timestamp(date).date()]
                series[date] = {
                    "x": [str(t)[:19] for t in day_df.index],
                    "y": day_df[col].where(day_df[col].notna(), None).tolist(),
                }
            layout = {
                "height": 260,
                "margin": {"l": 60, "r": 20, "t": 10, "b": 50},
                "xaxis": {"title": "Time"},
                "yaxis": {"title": col},
                "showlegend": False,
                "paper_bgcolor": "white",
                "plot_bgcolor": "#f7f9fc",
            }
            charts.append(
                {
                    "label": col,
                    "id": f"daily-{col.replace('/', '-')}",
                    "dates": dates,
                    "series_json": json.dumps(series),
                    "layout_json": json.dumps(layout),
                }
            )
        return charts

    def _collect_data(self) -> dict[str, Any]:
        """Collect all template variables for the tremor report.

        Returns:
            dict[str, Any]: Context dict containing completeness metrics,
                band statistics, and Plotly chart JSON strings.
        """
        df = self._tremor.df
        return {
            "station": self._station,
            "start_date": str(df.index[0].date()) if not df.empty else "—",
            "end_date": str(df.index[-1].date()) if not df.empty else "—",
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "completeness": self._completeness(df),
            "band_stats": self._band_stats(df),
            "overview_charts": self._overview_charts_by_band(df),
            "daily_charts": self._daily_charts_by_band(df),
        }
